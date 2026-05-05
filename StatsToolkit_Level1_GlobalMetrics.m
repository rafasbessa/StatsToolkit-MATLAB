function stats = StatsToolkit_Level1_GlobalMetrics(model)
% STATSTOOLKIT_LEVEL1_GLOBALMETRICS Computes global quality metrics for mixed models.
%
% Statistically rigorous implementation based on:
%   - Nakagawa et al. (2013, 2017)
%   - Johnson (2014)
%
% Syntax:
%   stats = StatsToolkit_Level1_GlobalMetrics(model)
%
% Inputs:
%   model - A fitted LinearMixedModel (LMM) or GeneralizedLinearMixedModel (GLMM) object.
%
% Outputs:
%   stats - Structure containing the following metrics:
%       .ModelType      : String indicating LMM or GLMM.
%       .Phi            : Dispersion parameter (1.0 for LMMs, estimated for GLMMs).
%       .R2_Marginal    : Variance explained by fixed effects only.
%       .R2_Conditional : Variance explained by fixed and random effects.
%       .ICC            : Intraclass Correlation Coefficient (repeatability).
%       .f2             : Cohen's f2 (global marginal effect size).
%       .AICc           : Corrected Akaike Information Criterion.
%
% Dependencies: Statistics and Machine Learning Toolbox (R2025a)

    % ==========================================
    % 1. Data Alignment & Base Extraction
    % ==========================================
    % Uses exactly the dataset processed internally by the model to prevent mismatches
    dataUsed = model.Variables;
    n = double(model.NumObservations);
    
    % Total number of estimated parameters (fixed + random + residual)
    % This correctly derives 'k' for AICc computation via mathematical identity.
    aic  = double(model.ModelCriterion{1, 'AIC'});
    logL = double(model.ModelCriterion{1, 'LogLikelihood'});
    k    = (aic + 2*logL) / 2;
    
    % Extract covariance components (residual variance)
    [~, sig2] = covarianceParameters(model);
    
    % ==========================================
    % 2. Model Type Differentiation & Fixed/Resid Variance
    % ==========================================
    if isa(model, 'LinearMixedModel')
        % ----------------------
        % LMM Handling
        % ----------------------
        stats.ModelType = 'LMM';
        stats.Phi = 1.0;  % Gaussian models assume phi = 1
        
        % Fixed-effect variance: Variance of X*beta (population-level predictions)
        yFix = predict(model, dataUsed, 'Conditional', false);
        var_f = var(double(yFix), 1); % Population variance (1st degree of freedom)
        
        % Residual variance directly estimated
        var_resid = double(sig2);
        
    elseif isa(model, 'GeneralizedLinearMixedModel')
        % ----------------------
        % GLMM Handling
        % ----------------------
        stats.ModelType = 'GLMM';
        
        % Dispersion parameter (Pearson Chi-square / df)
        pearsonRes = residuals(model, 'ResidualType', 'Pearson');
        stats.Phi = sum(pearsonRes.^2) / double(model.DFE);
        
        % Fixed-effect variance on latent (link) scale
        mu = predict(model, dataUsed, 'Conditional', false);
        eta = model.Link.Link(mu);
        var_f = var(double(eta), 1);
        
        % Mean on response scale (inverse-link) for Delta method approximations
        mu_bar = mean(double(mu), 'omitnan');
        dist = lower(model.Distribution);
        link = lower(model.Link.Name);
        
        % Distribution-specific residual variance (Nakagawa 2017)
        switch dist
            case 'binomial'
                % Latent-scale theoretical variance
                if strcmp(link, 'logit')
                    var_resid = (pi^2)/3;
                elseif strcmp(link, 'probit')
                    var_resid = 1;
                else
                    var_resid = (pi^2)/6; % cloglog approximation
                end
            case 'poisson'
                % Delta method (log link typical)
                if strcmp(link, 'log')
                    var_resid = log(1 + 1/mu_bar);
                else
                    var_resid = var(double(mu), 1, 'omitnan');
                end
            case {'gamma', 'inverse gaussian'}
                % Log-link delta approximation
                if strcmp(link, 'log')
                    var_resid = log(1 + stats.Phi);
                else
                    var_resid = stats.Phi;
                end
            case 'normal'
                var_resid = double(sig2);
            otherwise
                % Fallback
                var_resid = double(sig2);
        end
    else
        error('Model must be a LinearMixedModel or GeneralizedLinearMixedModel object.');
    end
    
    % ==========================================
    % 3. Random Effect Variance (Johnson 2014)
    % ==========================================
    % For random intercept-only: variance equals intercept variance.
    % For random slopes: total random variance equals the mean of ZGZ'
    % We approximate by computing variance of conditional modes (BLUPs) contribution.
    predCond = predict(model, dataUsed, 'Conditional', true);
    predMarg = predict(model, dataUsed, 'Conditional', false);
    
    if strcmp(stats.ModelType, 'LMM')
        % LMM: Response scale is the same as link scale
        reContrib = double(predCond - predMarg);
    else
        % GLMM: Bring both predictions to the LINK scale (eta) before subtracting, 
        % to obtain the pure latent variance of Zu
        etaCond = model.Link.Link(predCond);
        etaMarg = model.Link.Link(predMarg);
        reContrib = double(etaCond - etaMarg);
    end
    
    var_l = var(reContrib, 1, 'omitnan');
    
    % ==========================================
    % 4. Total Variance & Nakagawa R2
    % ==========================================
    total_var = var_f + var_l + var_resid;
    
    % Marginal R2: variance explained by fixed effects only
    stats.R2_Marginal = var_f / total_var;
    
    % Conditional R2: variance explained by fixed + random effects
    stats.R2_Conditional = (var_f + var_l) / total_var;
    
    % ==========================================
    % 5. ICC & Cohen's f2
    % ==========================================
    % ICC is strictly interpretable for random-intercept models, 
    % calculated here as a general repeatability measure.
    stats.ICC = var_l / (var_l + var_resid);
    
    % Cohen's f2 (Global Marginal Effect Size): f2 = R2 / (1 - R2)
    stats.f2 = stats.R2_Marginal / (1 - stats.R2_Marginal);
    
    % ==========================================
    % 6. Corrected AIC (AICc)
    % ==========================================
    % AICc = AIC + [2k(k+1)] / (n - k - 1)
    denominator = n - k - 1;
    if denominator > 0
        stats.AICc = aic + (2 * k * (k + 1)) / denominator;
    else
        % Fallback for overparameterized models where n is too small
        warning('Sample size too small for reliable AICc calculation. Returning AIC instead.');
        stats.AICc = aic;
    end
end