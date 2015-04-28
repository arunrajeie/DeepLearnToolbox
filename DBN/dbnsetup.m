function dbn = dbnsetup(dbn, x, opts)
    n = size(x, 2);
    dbn.sizes = [n, dbn.sizes];

    for u = 1 : numel(dbn.sizes) - 1
        dbn.rbm{u}.alpha    = opts.alpha;

        %%% Momentum Transitioning
        % From Hinton's Handbook, it is suggested that one should start
        % with an initial momemntum level, perhaps at around 0.5 or so,
        % and then after the learning has settled, to kick this up to 
        % about 0.9 or so.
        dbn.rbm{u}.momentum_initial = opts.momentum_initial;
        dbn.rbm{u}.momentum_change_epoch = opts.momentum_change_epoch;
        dbn.rbm{u}.momentum_final = opts.momentum_final;

        dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
