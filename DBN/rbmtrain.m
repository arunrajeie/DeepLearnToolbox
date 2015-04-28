function rbm = rbmtrain(rbm, x, opts)
    assert(isfloat(x), 'x must be a float');
    assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
    m = size(x, 1);
    numbatches = m / opts.batchsize;
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');

    for i = 1 : opts.numepochs
        kk = randperm(m);
        err = 0;

        %%% Momentum Transitioning
        if i < rbm.momentum_change_epoch
            rbm.momentum = rbm.momentum_initial;
        else
            rbm.momentum = rbm.momentum_final;
        end

        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
            
            v1 = batch;
            h1 = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W');
            v2 = sigmrnd(repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W);
            h2 = sigm(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W');

            c1 = h1' * v1;
            c2 = h2' * v2;

            rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * (c1 - c2)     / opts.batchsize;
            rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v1 - v2)' / opts.batchsize;
            rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h1 - h2)' / opts.batchsize;

            %%% Support for Weight Decay
            % According to Hinton's Handbook on training RBM, it is useful to add
            % some kind of weight decay, that is, to regularize the value of the
            % weights according to an $\ell_1$ or $\ell_2$ penalty.
            switch rbm.weight_decay
                case 'l2'
                    vW_pen = rbm.weight_cost .* rbm.W;
                    rbm.W = rbm.W + rbm.vW - rbm.alpha.*vW_pen;        
                case 'l1'
                    vW_pen = rbm.weight_cost .* sign(rbm.W);
                    rbm.W = rbm.W + rbm.vW - rbm.alpha.*vW_pen;        
                case 'none'
                    rbm.W = rbm.W + rbm.vW;        
            end            
            
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc;

            err = err + sum(sum((v1 - v2) .^ 2)) / opts.batchsize;
        end
        
        fprintf('[%d/%d] rec. err. : %0.3f | W range : %0.3e\n',i,opts.numepochs,err./numbatches,max(rbm.W(:))-min(rbm.W(:)));                

        if opts.visualize
            figure(1);
            subplot(1,2,1);
                visualize(rbm.W');
                colorbar();
                axis square;
            subplot(1,2,2);
            hold on;
                scatter(i,err./numbatches,'ro');
            hold off;
            axis square;
            box on;
            grid on;
            drawnow;
        end
    end
end
