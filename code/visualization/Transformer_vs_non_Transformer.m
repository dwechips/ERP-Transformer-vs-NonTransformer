enc_csv = resolve_file({ ...
    fullfile(pwd,'results','boot_meanr','meanr_bootstrap_summary.csv'), ...
    fullfile(pwd,'meanr_bootstrap_summary.csv')});
rsa_csv = resolve_file({ ...
    fullfile(pwd,'results','rsa_encoding_summary.csv'), ...
    fullfile(pwd,'rsa_encoding_summary.csv')});

enc = readtable(enc_csv);
rsa = readtable(rsa_csv);

enc = normalize_enc_cols(enc);
rsa = normalize_rsa_cols(rsa);

enc_model = pickcol(enc, {'Model','Name'}, enc.(enc.Properties.VariableNames{1}));
meanr_dir = pickcol(enc, {'Mean_r_direct','Mean_r_fromFile'}, []);
if isempty(meanr_dir)
    error('meanr_bootstrap_summary.csv missing Mean_r_direct or Mean_r_fromFile');
end
ci_lo = pickcol(enc, {'CI_low','Lower','Low','Lo'}, NaN(height(enc),1));
ci_hi = pickcol(enc, {'CI_high','Upper','High','Hi'}, NaN(height(enc),1));

rsa_model  = pickcol(rsa, {'Model','Name'}, rsa.(rsa.Properties.VariableNames{1}));
rsa_rho    = pickcol(rsa, {'RSA_rho','rho','Spearman','Spearman_rho'}, NaN(height(rsa),1));
rsa_p      = pickcol(rsa, {'RSA_p','p','pval','p_value'}, NaN(height(rsa),1));
median_enc = pickcol(rsa, {'Median_r_encoding','Median','Median_r'}, NaN(height(rsa),1));
mean_enc_r = pickcol(rsa, {'Mean_r_encoding','Mean_r'}, NaN(height(rsa),1));

M = table();
M.ModelKey = lower(strtrim(string(enc_model)));
M.Model    = string(enc_model);
M.Mean_r   = meanr_dir;
M.CI_low   = ci_lo;
M.CI_high  = ci_hi;

rsa_key = lower(strtrim(string(rsa_model)));
[tf,loc] = ismember(M.ModelKey, rsa_key);
M.Median_r = NaN(height(M),1);
M.RSA_rho  = NaN(height(M),1);
M.RSA_p    = NaN(height(M),1);
M.Median_r(tf) = median_enc(loc(tf));
M.RSA_rho(tf)  = rsa_rho(loc(tf));
M.RSA_p(tf)    = rsa_p(loc(tf));

bases_T  = ["llama2-7b","mistral-7b"];
bases_NT = ["rwkv-7b","stripedhyena-7b"];

sel_T  = select_variants(M.ModelKey, bases_T);
sel_NT = select_variants(M.ModelKey, bases_NT);

isT  = ismember(M.ModelKey, sel_T);
isNT = ismember(M.ModelKey, sel_NT);

meanr_T   = mean(M.Mean_r(isT),'omitnan');
meanr_NT  = mean(M.Mean_r(isNT),'omitnan');
CI_T      = [mean(M.CI_low(isT),'omitnan'),  mean(M.CI_high(isT),'omitnan')];
CI_NT     = [mean(M.CI_low(isNT),'omitnan'), mean(M.CI_high(isNT),'omitnan')];

median_T  = mean(M.Median_r(isT),'omitnan');
median_NT = mean(M.Median_r(isNT),'omitnan');

rho_T     = mean(M.RSA_rho(isT),'omitnan');
rho_NT    = mean(M.RSA_rho(isNT),'omitnan');

figure('Position',[100 100 1080 360]);

cats = categorical({'Transformer','Non-Transformer'});
cats = reordercats(cats, {'Transformer','Non-Transformer'}); 
xc = double(cats);

subplot(1,3,1);
bar(cats, [meanr_T, meanr_NT]); hold on;
errL = [meanr_T-CI_T(1),  meanr_NT-CI_NT(1)];
errU = [CI_T(2)-meanr_T,  CI_NT(2)-meanr_NT];
errorbar(xc, [meanr_T, meanr_NT], errL, errU, 'k.', 'CapSize',10, 'LineWidth',1.1);
ylabel('Mean r (encoding)');
title('Encoding (Mean r, 95% CI)');
set(gca,'TickLabelInterpreter','none'); 
box off;

subplot(1,3,2);
bar(cats, [median_T, median_NT]);
ylabel('Median r (encoding)');
title('Encoding (Median r)');
set(gca,'TickLabelInterpreter','none'); box off;

subplot(1,3,3);
bar(cats, [rho_T, rho_NT]);
ylabel('RSA \rho');
title('Representational Similarity');
set(gca,'TickLabelInterpreter','none'); box off;

sgtitle('Transformer vs Non-Transformer');
saveas(gcf,'transformer_vs_nontransformer.png');

T_enc = M(:,{'Model','Mean_r','CI_low','CI_high','Median_r'});
T_enc = [T_enc;
         {string('Transformer (avg)'),   meanr_T, CI_T(1), CI_T(2), median_T};
         {string('Non-Transformer (avg)'), meanr_NT, CI_NT(1), CI_NT(2), median_NT}];
writetable(T_enc,'table_encoding_with_groups.csv');
disp('Exported: table_encoding_with_groups.csv');

T_rsa = M(:,{'Model','RSA_rho','RSA_p'});
T_rsa = [T_rsa;
         {string('Transformer (avg)'),   rho_T,  mean(M.RSA_p(isT),'omitnan')};
         {string('Non-Transformer (avg)'), rho_NT, mean(M.RSA_p(isNT),'omitnan')}];
writetable(T_rsa,'table_rsa_with_groups.csv');
disp('Exported: table_rsa_with_groups.csv');

function fp = resolve_file(candidates)
    for i=1:numel(candidates)
        if isfile(candidates{i})
            fp = candidates{i}; return;
        end
    end
    error('missing files：\n  %s', strjoin(candidates, newline));
end

function col = pickcol(T, names, defaultVal)
    vars = string(T.Properties.VariableNames);
    if ischar(names) || isstring(names), names = cellstr(names); end
    idx = find(ismember(lower(vars), lower(string(names))), 1, 'first');
    if ~isempty(idx)
        col = T.(vars{idx});
    else
        col = defaultVal;
    end
end

function T = normalize_enc_cols(T)
    vn = string(T.Properties.VariableNames);
    T = rename_if_exists(T, vn, "model","Model");
    T = rename_if_exists(T, vn, "mean_r_direct","Mean_r_direct");
    T = rename_if_exists(T, vn, "mean_r_fromfile","Mean_r_fromFile");
    T = rename_if_exists(T, vn, "ci_low","CI_low");
    T = rename_if_exists(T, vn, "ci_high","CI_high");
end

function T = normalize_rsa_cols(T)
    vn = string(T.Properties.VariableNames);
    T = rename_if_exists(T, vn, "model","Model");
    T = rename_if_exists(T, vn, "rsa_rho","RSA_rho");
    T = rename_if_exists(T, vn, "rsa_p","RSA_p");
    T = rename_if_exists(T, vn, "mean_r_encoding","Mean_r_encoding");
    T = rename_if_exists(T, vn, "median_r_encoding","Median_r_encoding");
end

function T = rename_if_exists(T, vn, from, to)
    hit = find(strcmpi(vn, from), 1);
    if ~isempty(hit) && ~ismember(to, vn)
        T = renamevars(T, vn(hit), to);
    end
end

function sel = select_variants(modelKeys, bases)
    keys = string(modelKeys(:));
    sel = strings(0,1);
    for b = string(bases(:))'
        cand = keys(startsWith(keys, lower(b)));
        if isempty(cand), continue; end
        hit = [];
        ord = ["-original_sentences", "-original", ""];
        for suf = ord
            hit = cand(contains(cand, lower(suf)));
            if ~isempty(hit), sel(end+1,1) = hit(1); break; end
        end
        if isempty(hit)
            sel(end+1,1) = cand(1);
        end
    end
end
