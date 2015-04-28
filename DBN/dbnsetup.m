function dbn = dbnsetup(dbn, x, opts)
    n = size(x, 2);
    dbn.sizes = [n, dbn.sizes];

    for u = 1 : numel(dbn.sizes) - 1
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

        % According to page 9 of 
        %    G. Hinton, ``A practical Guide To Training RBMs'' 2010,        
        % it is desirable to draw the initial weightings from a normal
        % distribution of zero mean and standard deviation of 0.01.
        dbn.rbm{u}.W  = 0.01.*randn(dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
