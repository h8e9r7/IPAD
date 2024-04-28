
import time
import torch
from modules.utils.metric import AverageMeter


def do_train(cfg, model, data_loader, gt, lt, optimizer, intra_loss, inter_loss, proto_loss, vce_loss, device, logger, epoch, mem, mem2):
    losses = AverageMeter()
    if lt is not None:
        glosses = AverageMeter()
        llosses = AverageMeter()
        interlosses = AverageMeter()
    # print('Freeze Stage 2')
    scalar = torch.cuda.amp.GradScaler()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    end = time.time()
    for idx, batch in enumerate(data_loader):
        with torch.cuda.amp.autocast():
            x, p, n, a, flag, deflag, x_a, p_a, n_a, x_v, p_v, n_v, x_id, p_id, n_id = batch
            #x is list
            n_data = len(x)
            a = a.to(device)
        
            flag = flag.to(device)
            deflag = deflag.to(device)
            
            gxx = torch.stack([gt(i) for i in x], dim=0).to(device) #gx[B,C,H,W]
            gxx1 = torch.stack([gt(i, transform_type=1) for i in x], dim=0).to(device)
            gxx2 = torch.stack([gt(i, transform_type=2) for i in x], dim=0).to(device)
            
            gpp = torch.stack([gt(i) for i in p], dim=0).to(device)
            gnn = torch.stack([gt(i) for i in n], dim=0).to(device)

            data_time.update(time.time() - end)

            # equal 0 means same attribute
            a_flag = torch.cat([a,a,a],dim=0)
            a_row = a_flag.unsqueeze(-1)
            a_column = a_flag.unsqueeze(0)
            a_flag = a_row - a_column
            a_flag = a_flag.masked_fill(a_flag != 0., float(-10000.0)) 

            
            # cluster_loss
            
            if lt is None: #stage1
                gx, gx_attnmap = model(gxx, a, level='global')
                gp, gp_attnmap = model(gpp, a, level='global')
                gn, gn_attnmap = model(gnn, a, level='global')

                gx1, _ = model(gxx1, a, level='global')
                gx2, _ = model(gxx2, a, level='global')

                """
                Uncomment this block to enable MemoryBank
                """
                closs = 0
                mem.update(gx, x_id, x_a, x_v)
                mem.update(gp, p_id, p_a, p_v)
                mem.update(gn, n_id, n_a, n_v)
                # print('mem full count:')
                # print(mem.get_prototype_count())
                
                proto_x = mem(x_a, x_v)
                if proto_x is not None:
                    proto_p = mem(p_a, p_v)
                    proto_n = mem(n_a, n_v)
                    closs = proto_loss(gx, proto_x[0], proto_x[1]) + \
                            proto_loss(gp, proto_p[0], proto_p[1]) + \
                            proto_loss(gn, proto_n[0], proto_n[1]) / 3.
                else:
                    pass
                    # print('warm up')
                
                intra = cfg.SOLVER.REGION_WEIGHT * \
                    (intra_loss(gx,gp,gn)+intra_loss(gx1,gp,gn)+intra_loss(gx2,gp,gn)+intra_loss(gx,gx1,gn)+intra_loss(gx,gx2,gn)) / 5.
                loss = intra + closs

                # print("x_a", x_a, "x_v", x_v)
                vce = vce_loss(model.vmlp(gx, x_a), x_v)
                # print(f"vce: {vce}")
                loss += 0.1 * vce

                # print(f"loss: {loss}, intra: {intra}, closs: {closs}, vce: {vce}")
            else:#stage2
                # 如果global比较高就把它冻结
                # with torch.no_grad():
                gx, gx_hat, gx_attnmap = model(gxx, a, level='global')
                gp, gp_hat, gp_attnmap = model(gpp, a, level='global')
                gn, gn_hat, gn_attnmap = model(gnn, a, level='global')

                gx1, _, _ = model(gxx1, a, level='global')
                gx2, _, _ = model(gxx2, a, level='global')

                loss = cfg.SOLVER.REGION_WEIGHT * \
                    (intra_loss(gx,gp,gn)+intra_loss(gx1,gp,gn)+intra_loss(gx2,gp,gn)+intra_loss(gx,gx1,gn)+intra_loss(gx,gx2,gn)) / 5.
                glosses.update(loss.cpu().item(), n_data)


                """
                Uncomment this block to enable MemoryBank
                """
                closs = 0
                mem.update(gx, x_id, x_a, x_v)
                mem.update(gp, p_id, p_a, p_v)
                mem.update(gn, n_id, n_a, n_v)
                # print('mem full count:')
                # print(mem.get_prototype_count())
                
                proto_x = mem(x_a, x_v)
                if proto_x is not None:
                    proto_p = mem(p_a, p_v)
                    proto_n = mem(n_a, n_v)
                    closs = proto_loss(gx, proto_x[0], proto_x[1]) + \
                            proto_loss(gp, proto_p[0], proto_p[1]) + \
                            proto_loss(gn, proto_n[0], proto_n[1]) / 3.
                else:
                    pass
                    # print('warm up')
                loss += closs
                

                gx_attnmap = gx_attnmap.cpu().detach().numpy()
                gp_attnmap = gp_attnmap.cpu().detach().numpy()
                gn_attnmap = gn_attnmap.cpu().detach().numpy()

                lx = torch.stack([lt(i, mask) for i, mask in zip(x, gx_attnmap)], dim=0).to(device)
                lp = torch.stack([lt(i, mask) for i, mask in zip(p, gp_attnmap)], dim=0).to(device)
                ln = torch.stack([lt(i, mask) for i, mask in zip(n, gn_attnmap)], dim=0).to(device)
                # with torch.no_grad():
                lx, lx_attmap = model(lx, a, level='local')
                lp, lp_attmap = model(lp, a, level='local')
                ln, ln_attmap = model(ln, a, level='local')


                closs2 = 0
                mem2.update(lx, x_id, x_a, x_v)
                mem2.update(lp, p_id, p_a, p_v)
                mem2.update(ln, n_id, n_a, n_v)
                proto2_x = mem2(x_a, x_v)
                if proto2_x is not None:
                    proto2_p = mem2(p_a, p_v)
                    proto2_n = mem2(n_a, n_v)
                    closs2 = proto_loss(lx, proto2_x[0], proto2_x[1]) + \
                            proto_loss(lp, proto2_p[0], proto2_p[1]) + \
                            proto_loss(ln, proto2_n[0], proto2_n[1]) / 3.
                else:
                    pass
                loss += closs2


                # local losses
                l = intra_loss(lx, lp, ln)
                llosses.update(cfg.SOLVER.PATCH_WEIGHT * l.cpu().item(), n_data)
                loss += cfg.SOLVER.PATCH_WEIGHT * l
                
                pp = torch.cat([gx,gp,gn],dim=0)
                xx = torch.cat([lx,lp,ln],dim=0)
                nn = torch.cat([gx_hat,gp_hat,gn_hat],dim=0)
                            
                interloss = cfg.SOLVER.INTER_WEIGHT * inter_loss(pp, xx, nn, flag, deflag, a_flag, T = cfg.SOLVER.TAU, alpha = cfg.SOLVER.ALPHA, type=1)

                interlosses.update(interloss.cpu().item(), n_data)
                loss +=  interloss

                # vce = vce_loss(model.vmlp(lx, x_a), x_v) + \
                #       vce_loss(model.vmlp(lp, p_a), p_v) + \
                #       vce_loss(model.vmlp(ln, n_a), n_v)
                # # print(f"vce: {vce}")
                # loss += 0.1 * vce
                
                # print(f"loss: {loss}, closs: {closs}, closs2: {closs2}, interloss: {interloss}")
                stage_3 = False
                if stage_3:
                    # # print(vx.shape) # batch * 1024
                    vx, lx_v_pred = model.value_forward(lx, lx_attmap, x_a)
                    vp, lp_v_pred = model.value_forward(lp, lp_attmap, p_a)
                    vn, ln_v_pred = model.value_forward(ln, ln_attmap, n_a)
                    # print(x_a, x_v, lx_v_pred)
                    # print(p_a, p_v, lp_v_pred)
                    # print(n_a, n_v, ln_v_pred)
                    vloss = 0.
                    closs3 = 0.
                    proto2_x = mem2(x_a, x_v)
                    if proto2_x is not None:
                        proto2_p = mem2(p_a, p_v)
                        proto2_n = mem2(n_a, n_v)
                        closs3 = proto_loss(lx, proto2_x[0], proto2_x[1]) + \
                                proto_loss(lp, proto2_p[0], proto2_p[1]) + \
                                proto_loss(ln, proto2_n[0], proto2_n[1]) / 3.
                        vloss = intra_loss(vx, mem2(x_a, x_v)[0], vn)
                    print(vloss, closs3)
                    loss += vloss + closs3

                    
        losses.update(loss.cpu().item(), n_data)

        optimizer.zero_grad()

        # loss.backward()
        # optimizer.step()
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()


        batch_time.update(time.time() - end)
        end = time.time()


        local_log = (f"Regin-intra Loss: {glosses.val:.4f}({glosses.avg:.4f})\t"+\
                    f"Inter Loss: {interlosses.val:.4f}({interlosses.avg:.4f})\t"+\
                    f"Patch-intra Loss: {llosses.val:.4f}({llosses.avg:.4f})\t") if lt is not None else ""
        if idx % cfg.SOLVER.LOG_PERIOD == 0:
            logger.info(f"Train Epoch: [{epoch}][{idx}/{len(data_loader)}]\t"+
                        local_log+
                         f"Loss: {losses.val:.4f}({losses.avg:.4f})\t"+
                         f"Batch Time: {batch_time.val:.3f}({batch_time.avg:.3f})\t"+
                         f"Data Time: {data_time.val:.3f}({data_time.avg:.3f})")
            
    return (losses.avg, glosses.avg, llosses.avg, interlosses.avg) if lt is not None else losses.avg



