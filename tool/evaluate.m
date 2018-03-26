addpath(genpath('/home/zhangyu/codes/supervision/evaluate/'));
datasets = {'ECSSD', 'PASCAL-S', 'BSD', 'DUT-O'};
data_root = '/disk2/zhangyu/data/saliency/';
ckp_epoch = 30;
predict_root = ['../data/results/epoch' num2str(ckp_epoch) '/'];
ext = '.png';

for idx = 1:length(datasets)
    dataset = datasets{idx};
    img_dir = [data_root dataset '/images/'];
    mask_dir = [data_root dataset '/GT/'];
    predict_dir = [predict_root dataset '/'];
    performance = evaluate_SO(dataset, img_dir, mask_dir, predict_dir, ext)
    save([predict_root dataset '_result.mat'], 'performance')
end
