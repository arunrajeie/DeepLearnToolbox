function test_example_DBN
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rand('state',0)
dbn.sizes = [100];
opts.numepochs =   10;
opts.weight_decay = 'l2';
opts.weight_cost = 0.01;
opts.batchsize = 100;
opts.momentum_initial = 0.5;
opts.momentum_final = 0.9;
opts.momentum_change_epoch = 20;
opts.alpha     =   0.0005;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
dbn.sizes = [100 100];
opts.numepochs =   10;
opts.weight_decay = 'l2';
opts.weight_cost = 0.01;
opts.batchsize = 100;
opts.momentum_initial = 0.5;
opts.momentum_final = 0.9;
opts.momentum_change_epoch = 20;
opts.alpha     =   0.0005;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  1;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.10, 'Too big error');
