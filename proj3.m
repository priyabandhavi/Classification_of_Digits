function [] = proj3(TrainImages,TrainLabels,TestImages,TestLabels)
[X,T] = preprocess(TrainImages,TrainLabels);
[W_lr]= train_lr(X,T);
test_lr(TestImages,TestLabels,W_lr);
train_nn(X,T);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[X,T] = preprocess(h1,h2)
images = loadMNISTImages(h1);
labels = loadMNISTLabels(h2);
X = images';
T = zeros(60000,10);

    for i=1:60000
        r = labels(i);
        T(i,r+1)= 1; 
       
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
%images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[W]= train_lr(x,y)

input = x;
target = y;

blr = zeros(1,10);

W = ones(784,10);
iteration = 630;
eeta = 0.0001;

for i=1:iteration
    mult=input*W; %+ bl(1:24000,:) ;
    topNum= exp(mult);
    botDenom=sum(topNum,2);    
    [row,col]=size(mult);
    Y=zeros(row,col);
    for a=1: row
        for b=1:col
            Y(a,b)= topNum(a,b)/botDenom(a);
        end
    end
       
    grad=input'*(Y - target);
    W=W- eeta*grad;
   
end
Wlr = W;
%save('variables.mat','W','-append');
save('proj3.mat','Wlr','blr');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[] = test_lr(t1,t2,wt)
X_test = t1;
T_test = t2;
W = wt;
topNum= exp(X_test*W); %+ bl(24001:30000,:));
botY=sum(topNum,2);

[row,col]=size(X_test*W);
Y_test=zeros(row,col);
for a=1: row
    for b=1:col
        Y_test(a,b)= topNum(a,b)/botY(a);
    end
end

[B,D2] = max(Y_test,[],2);
[B1,D4]= max(T_test,[],2);
count=0;
for i=1:row
    if (D2(i)==D4(i))
        count=count+1;
    end
end
err = count/length(X_test);
fprintf('accuracy is %4.2f\n',err);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ out ] = softmax(r)
out = exp(r)./sum(exp(r));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ out ] = logsig( n )
out = 1./(1 + exp(-1.*n));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[] = train_nn(X,t)
Data = X;
T = t;
[N,L]=size(Data);
in = Data(:,2:end);
train_rate = 0.0903;
in = [ones(N,1) in];
D = 784;
K = 10; 
M = 70;
%load('originals.mat');
Wt1 = rand(M,D)*2;
 Wt2 = rand(K,M)*2;
wt1 = ones(M,D) - Wt1;
wt2 = ones(K,M) - Wt2;

for j = 1 : 1 : N
    i = round(rand(1)*N);
    if i==0
        i = i+1;
    end
    BJ = wt1*in(i,:)';
    ZJ = logsig(BJ);
    KA = wt2*ZJ;
    XD = softmax(KA);
    kat = XD' - T(i,:);
    Gtwo = kat' * ZJ';
    Diff = (ZJ .* (ones(M,1) - ZJ)) .* (wt2'*kat');
    Gone = Diff*in(i,:);
    wt1 = wt1 - train_rate*Gone;
    wt2 = wt2 - train_rate*Gtwo;
end
bnn1 = zeros(1,70);
bnn2 = zeros(1,10);
h = 'sigmoid';
Wnn1 = transpose(wt1);
Wnn2 = transpose(wt2);
save('proj3.mat','bnn1','bnn2','h','Wnn1','Wnn2','-append');
end


