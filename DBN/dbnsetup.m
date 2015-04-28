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

        if u==1
            %%% Set Initial Visible Bias
            % According to Hinton's RBM handbook, the initial visible bias
            % should be set according to the log-ratio of the 1/0 probabilities
            % of the training set.
            p = mean(x,1);                          % Estimate P[x_i==1]
            % We clip the probabilities to prevent Inf and NaN errors...
            p(p > 1-(1e-8)) = 1-(1e-8);             % Clip upper value of probabilities
            p(p < 1e-8) = 1e-8;                     % Clip lower value of probabilities
            % Final bias assignment according to log probabilities
            dbn.rbm{u}.b = log(p./(1-p))';        
        else
            % If this is the "lower" biasing for a deeper layer of the
            % stacked RBM, then we should intialize the bias on this 
            % hidden layer to zero.
            dbn.rbm{u}.b = zeros(dbn.size(u), 1);
        end
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
