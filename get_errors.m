function [angle_decoding] = get_errors(testdata_angle,preddata_angle)
   
%% Get Errors
    ymean_angle = mean(testdata_angle);
        sse = sum((testdata_angle-preddata_angle).^2);
        sst = sum((testdata_angle -ymean_angle).^2);
        R2_angle = 1-(sse/sst);   
    temp = corrcoef(testdata_angle,preddata_angle);
    corr_coeff_angle = temp(1,2);
    mean_err_angle = sum(abs(preddata_angle-testdata_angle))/length(preddata_angle);
    
    angle_decoding = [R2_angle corr_coeff_angle mean_err_angle];
end


