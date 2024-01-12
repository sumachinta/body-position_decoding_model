function [testdata_feature,preddata_feature,indices] = performdecoding(trial_start_idx,trial_end_idx,X,output_feature,indices)
%% Perform decoding
testdata_feature = []; preddata_feature = []; 
% k-fold cross validation
if ~exist('indices','var')
    indices = crossvalind('Kfold',size(trial_start_idx,1),15);
end
for k = 1:length(unique(indices))
    clear id;id = indices==k;

% Separate to training and test data (based on trials)
    clear st;st = trial_start_idx(id==1,1);
    clear sp;sp = trial_end_idx(id==1,1);
    test_id = [];
    for i = 1:sum(id)
        test_id = [test_id st(i):sp(i)];
    end
    train_id=[]; train_id = 1:size(X,1); train_id(test_id)=[];

    clear Xtest; clear Xtrain; clear ytrain; clear ytest;
    Xtest = X(test_id,:);
    Xtrain = X(train_id,:);
    
    % Angle
    ytrain_feature = output_feature(train_id,:);
    ytest_feature  = output_feature(test_id,:);
    mdl_feature = fitlm(Xtrain,ytrain_feature);%,'Exclude',ytrain==0);
    clear ypred_feature; ypred_feature = predict(mdl_feature,Xtest);
   
    out = mat2cell(ytest_feature', 1, (sp-st)+1);
    outp = mat2cell(ypred_feature', 1, (sp-st)+1);
    bin_res = .015;
    %% Plot all real vs predicted data
    % figure
    % a = trial_start_idx(id);
    % for i = 1:length(out) % breaking into trials
    %     x = 0:bin_res:(size(out{1,i},2)-1)*bin_res;
    %     subplot(sum(id),1,i)
    %     % figure
    %     plot(x,smoothdata(out{1, i},2,'movmean',5),'k-')
    %     % plot(x,out_{1, i},'k-')
    %     hold on
    %     plot(x,smoothdata(outp{1, i},2,'movmean',5),'b-')
    %     % plot(x,smoothdata(outp_{1, i},1,'movmean',5),'b-')
    %     % xlabel('Time (sec)')
    %     ylabel('Real')
    %     ylabel('Predicted')
    %     % title([num2str(a(i))]);
    %     axis tight; box off; %axis square;
    %     title(a(i))
    %     ylim([prctile(output_feature,.5) prctile(output_feature,99.5)])
    % end

testdata_feature = [testdata_feature; ytest_feature];
preddata_feature = [preddata_feature; ypred_feature];
end
%% testing on a randome trial
tt = randi(length(trial_start_idx),[1,1]);%find(trial_start_idx==16998);
clear st;st = trial_start_idx(tt,1);
clear sp;sp = trial_end_idx(tt,1);
test_id = []; test_id = [st:sp];
clear Xtest; clear ytest;
Xtest = X(test_id,:);
ytest_feature  = output_feature(test_id,:);

ypred_feature = predict(mdl_feature,Xtest);   
figure
x = 0:bin_res:(size(ypred_feature,1)-1)*bin_res;
% plot(x,ypred_feature,'b-'); hold on
% plot(x,ytest_feature,'k-');
plot(x,smoothdata(ypred_feature,1,'movmean',5),'b-'); hold on
plot(x,smoothdata(ytest_feature,1,'movmean',5),'k-');
box off;
legend('Predicted','Real');
xlabel('time(s)');
ylabel('feature');
end
