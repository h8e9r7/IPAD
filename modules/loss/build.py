from ast import If
import torch
import torch.nn as nn


def build_loss(cfg):
    return TripletRankingLoss(cfg)


class TripletRankingLoss(nn.Module):
    def __init__(self, cfg):
        super(TripletRankingLoss, self).__init__()
        self.margin = cfg.SOLVER.MARGIN
        self.device = torch.device(cfg.DEVICE)
        self.criterion = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, ref, pos, neg):
        x1 = nn.functional.cosine_similarity(ref, pos, dim=1)
        x2 = nn.functional.cosine_similarity(ref, neg, dim=1)
        # x1 = torch.sum(ref * pos, dim=1)
        # x2 = torch.sum(ref * neg, dim=1)
        
        target = torch.FloatTensor(ref.size(0)).fill_(1)
        target = target.to(self.device)
        loss = self.criterion(x1, x2, target)

        return loss
    

class PrototypeTripletRankingLoss(nn.Module):
    def __init__(self, cfg):
        super(PrototypeTripletRankingLoss, self).__init__()
        self.margin = cfg.SOLVER.PROTO_MARGIN
        self.device = cfg.DEVICE

    def forward(self, ref, pos, neg_list):
        """
        Calculate the contrastive loss for a batch of anchor-positive-negative triplets
        1 anchor : 1 positive : K_a-1 negative under the same attribute a
        L_v(I, a|v) = 1/(K_a - 1) * sum(max(0, m + d(I, a|v) - d(I, a|v')))
        :param ref: anchor feature, Tensor of shape [batch, embed_dim]
        :param pos: positive feature, Tensor of shape [batch, embed_dim]
        :param neg_list: a list of negative features, each Tensor of shape [K_a-1, embed_dim]
        """
        # calculate the cosine similarity between the anchor and the positive sample
        pos_similarity = nn.functional.cosine_similarity(ref, pos, dim=1)

        # initialize loss
        loss = 0.0

        # for each anchor and its corresponding positive and negative sample pairs, 
        # calculate the mean contrastive loss
        for i, anchor in enumerate(ref):
            # calculate the cosine similarity between the anchor and the negative sample
            neg_similarities = nn.functional.cosine_similarity(anchor.unsqueeze(0), neg_list[i], dim=1)
            # using clamp(x, min, max) to clamp the cosine similarity to the range [min, max]
            # clamp(x, min, max) 将输入张量每个元素的夹紧到区间 [min, max]，并返回结果到一个新张量
            triplet_loss = torch.mean(torch.clamp(self.margin + neg_similarities - pos_similarity[i], min=0.0))
            loss += triplet_loss

        # mean loss
        loss /= len(ref)

        return loss


class ValueCrossEntropy(nn.Module):
    def __init__(self, cfg):
        super(ValueCrossEntropy, self).__init__()
        self.device = cfg.DEVICE
    
    def forward(self, y_pred, value):
        # print(y_pred)
        loss = 0.
        
        for _o, _v in zip(y_pred, value):
            _loss = nn.functional.cross_entropy(_o.unsqueeze(0), torch.tensor([_v], dtype=torch.long).to(self.device))
            # print(_loss)
            loss += _loss

        loss /= len(value)

        return loss


class DELoss(nn.Module):
    def __init__(self):
        super(DELoss, self).__init__()

    def forward(self, xx, pp, nen, flag, deflag, a_flag,T=0.07,alpha=48.,type=1):
        '''
        FLAG [B,B]
        '''
        xx = nn.functional.normalize(xx, dim=1) #[B,C]
        pp = nn.functional.normalize(pp, dim=1) #[B,C]
        nen = nn.functional.normalize(nen, dim=1) #[B,C]

        singlex = torch.matmul(xx, xx.t()) # B B one - more
        xp = torch.matmul(xx, pp.t()) # B B one - more
        
        pos1 = singlex * torch.eye(singlex.shape[0])[:,:].cuda()
        # print(nominator.shape)
        pos1 = pos1.sum(dim=1).unsqueeze(1)
        pos = xp + flag
        neg1 = xp + deflag
        neg = torch.matmul(xx, nen.t()) # B B
        neg = neg + a_flag # B B

        pos = pos/T
        neg1 = neg1/T
        neg = neg/T
       
        nominator = torch.logsumexp(pos, dim=1)
        # print(nominator.shape)
        if type==1:
            de1 = torch.exp(torch.cat((pos,neg1),dim=1))
            de2 = alpha*torch.exp(neg)
            denominator = torch.log(torch.sum(torch.cat((de1,de2),dim=1),dim=1))
            
            # print(denominator.shape)
            # denominator = torch.cat((pos, neg1, neg), dim=1)
        elif type==2:
            de1 = torch.exp(pos)
            de2 = alpha*torch.exp(neg)
            denominator = torch.log(torch.sum(torch.cat((de1,de2),dim=1),dim=1))            
        elif type==3:
            nominator = torch.logsumexp(pos1, dim=1)
            de1 = torch.exp(torch.cat((pos1,neg1),dim=1))
            de2 = alpha*torch.exp(neg)
            denominator = torch.log(torch.sum(torch.cat((de1,de2),dim=1),dim=1))
        elif type==4:
            nominator = torch.logsumexp(pos1, dim=1)
            de1 = torch.exp(pos1)
            de2 = alpha*torch.exp(neg)
            denominator = torch.log(torch.sum(torch.cat((de1,de2),dim=1),dim=1))

        # denominator = torch.cat((pos, neg), dim=1)
        # print(denominator.shape)
        # denominator = torch.logsumexp(denominator, dim=1)
        return torch.mean(denominator - nominator)
