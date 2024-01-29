function [ab, slope, CI, m, global_std] = PLR (xx, yy, nstick, SL)
% This code is a modification of Guido Albertin's Piecewise Linear least
% Square Fit: (http://www.mathworks.com/matlabcentral/fileexchange/40913-piecewise-linear-least-square-fit)
%
%
% PLR, which stands for piecewise linear regression, is a function that fits
% a line consisting of connected straight sections to a set of data points
% using FMINSEARCH to minimise the total least squares error of all
% segments.
%
%
% This function has the added feature of determining the statistical significance
% of the slopes determined, given a certain number of sticks and significance
% level. This is done by performing ANOCOVA and Multiple Comparison
% Procedure.
%
%
% Input arguments:  xx and yy -> vectors of x- and y-coordinates respectively 
%                   of the input dataset.
%                   nsick -> number of connected straight sections.
%                   SL -> significance level (in %) to be used in the 
%                   Multiple Comparison Procedure.
% Output arguments: ab(:, 1) -> x-coordinates of endpoints and breakpoints 
%                   in ascending order.
%                   ab(:, 2) -> corresponding y-coordinates of endpoints 
%                   and breakpoints.
%                   CI -> confidence intervals for robust regression (i.e.
%                   when nstick = 1)
%                   m -> multiple comparison procedure results (only when
%                   nstick > 1) with second column containing slope's CIs
%                   and not standard error.
%                   global_std -> standard deviation after fitting piecewise 
%                   line (only when nstick > 1)

%----------------------------------------------------------------------O
% Initial Checks

if numel(xx) ~= numel(yy),
    error('ERROR: XX and YY are not equally long.')
end
if ~all(isfinite(xx)) || ~all(isfinite(yy)),
    warning('ERROR: XX or YY contain non-finite elements. *Non-finite elements removed assuming same index for non-finite elements')
    xx(~isfinite(xx))= []; yy(~isfinite(yy)) = [];
end
if numel(xx) < nstick + 1,
    error('ERROR: too few data points.')
end


%----------------------------------------------------------------------O
% Initialize coordinates of breakpoints.

ab        = zeros(nstick+1,2);
ab(1,1)   = min(xx);           ab(1,2)   = yy(xx==min(xx));
ab(end,1) = max(xx);           ab(end,2) = yy(xx==max(xx));
n = numel(ab);

if n > 4
    ab(:,1) = linspace(min(xx),max(xx),numel(ab)/2);
    ab(:,2) = interp1([ab(1,1), ab(end,1)],[ab(1,2) ab(end,2)], ab(:,1));
end

CI = []; % initialize confidence interval.

if nstick == 1, % Bi-square weighted robust regression for nstick = 1
    [pp, statsrob] = robustfit(xx, yy);
    ab(:, 2) = pp(2) * ab(:, 1) + pp(1);
    CI = [pp(2) - tinv(SL/100,numel(xx))*statsrob.ols_s/sqrt(numel(xx)), ...
        pp(2) + tinv(SL/100,numel(xx))*statsrob.ols_s/sqrt(numel(xx))]; % Display CI of robust regression
    slope = pp(2);
    m = [];
    global_std = statsrob.ols_s;
    
else % If nstick > 1, find the breakpoints that minimize the residuals.
    
    % reshape the ab matrix into a vector and delete the first and last
    % x-coord so that it is not included in optimization
    ab1 = reshape(ab,n,1); ab1(1) = []; ab1(nstick) = [];
    [ab1 fval] = fminsearch(@(ab1) MinimizedResuduals (ab1,xx,yy,n),ab1);
    
    global_std = fval;
    
    % reshape the optimized ab1 vector back to the ab matrix form and
    % compute slope
    ab = [min(xx), ab1(nstick); ab1(1:nstick-1), ab1(nstick+1:end-1); max(xx), ab1(end)];
    for j = 1:nstick
        slope(j) = (ab(j+1,2)-ab(j,2))/(ab(j+1,1)-ab(j,1));
    end

    % Analysing the statistical difference between slopes
    
    % First apply ANCOVA
    group = [];
    for i = 1:nstick
        group = [group; repmat(strjoin({'Stick ', num2str(i)}),numel(xx(xx>=ab(i,1) & xx<ab(i+1,1))),1)];
    end
    
    group = [group; strjoin({'Stick ', num2str(i)})];
    [h,atab,ctab,stats] = aoctool(xx,yy,group,'','','','','off');
    
    % Then perform Multiple Comparison Procedure
    figure;
    [~,m,~] = multcompare(stats,'Alpha',(1-SL/100),'display','on');
    % overwrite standard error in m matrix (the second column) with half of
    % CI
    m(:,2) = tinv(SL/100,numel(xx))*m(:,2)./sqrt(numel(xx));
end

%----------------------------------------------------------------------O
% Objective function for FMINSEARCH

    function resid = MinimizedResuduals(ab1,xx,yy,n)
        
        regress_trial = interp1([min(xx); ab1(1:nstick-1); max(xx)],ab1(nstick:end),xx);
        resid = double(sum((yy - regress_trial).^2));
        
    end

end
