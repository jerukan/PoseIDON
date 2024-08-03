%% Barrel generation
% units are in ft
BAR_R = 1.875 / 2;
BAR_H = 2.792;
BAR_VOL = pi * BAR_H * BAR_R^2;
% parameters
trials = 1000;
N_u = 500; % points to generate per barrel
N_cap_u = 60; % points for each end of the cylinder

lowest_depth = -2.1;
hor_lim = 5;

% metrics on how well the cylinder fit
% mean point cloud distance to cylinder
mean_distances = [];
% how close the cylinder axial vector is correct
orientation_scores = [];
cyl_models = {};
cyl_pcs = {};

for i=1:trials
    N = floor(normrnd(N_u, 100));
    N_cap = floor(normrnd(N_cap_u, 15));
    % barrel orientation
    bar_botxy = unifrnd(-hor_lim, hor_lim, 1, 2);
    bar_botz = unifrnd(lowest_depth, 0);
    bar_bot = [bar_botxy, bar_botz];
    % this is definitely not a uniform distribution
    % of orientations
    bar_vec = unifrnd(-1, 1, 1, 3);
    if bar_vec(3) < 0
        bar_vec(3) = -bar_vec(3);
    end
    bar_top = bar_bot + BAR_H * bar_vec / norm(bar_vec);

    % generation
    axialvec = bar_top - bar_bot;
    axialvec = axialvec / norm(axialvec);
    axialpoints = bar_bot + (bar_top - bar_bot) .* rand(N, 1);
    circr = BAR_R + normrnd(0, 0.025, N, 1);
    circtheta = rand(N, 1) * 2 * pi;
    circpoints = [cos(circtheta) .* circr, sin(circtheta) .* circr];
    axnull = null(axialvec);
    points = axialpoints + circpoints * axnull.';
    
    % Cylinder top/bottom generation for each end of the cylinder
    circr_cap = sqrt(rand(N_cap * 2, 1)) * BAR_R;
    circtheta_cap = rand(N_cap * 2, 1) * 2 * pi;
    circpoints_cap = [cos(circtheta_cap) .* circr_cap, sin(circtheta_cap) .* circr_cap];
    % append to cylinder edge points
    points = [
        points;
        bar_bot + circpoints_cap(1:N_cap, :) * axnull.';
        bar_top + circpoints_cap(N_cap:N_cap*2, :) * axnull.'
    ];

    % filter out buried points
    points_all = points;
    points = points(points(:, 3) >= 0, :);

    ptCloud_all = pointCloud(points_all);
    ptCloud = pointCloud(points);
    
    roi = [-inf inf; -inf inf; -inf inf];
    sampleIndices = findPointsInROI(ptCloud, roi);

    % fitting
    maxDistance = 0.25;
    [model, inlierIndices, ~, meanError] = pcfitcylinder( ...
        ptCloud, ...
        maxDistance, ...
        SampleIndices=sampleIndices ...
    );

    % correcting to barrel measurements
    model_p1 = model.Parameters(1:3);
    model_p2 = model.Parameters(4:6);
    % invalid cylinder generated, probably not enough points above
    % ground
    if all(model_p1 == model_p2)
        continue
    end
    if model_p1(3) > model_p2(3)
        above = model_p1;
        axvec = model_p2 - above;
    else
        above = model_p2;
        axvec = model_p1 - above;
    end
    axvec = axvec / norm(axvec);
    corr_below = above + BAR_H * axvec;
    model_new = cylinderModel([corr_below, above, BAR_R]);

    dists = evalCylinder(model_new.Parameters, ptCloud_all.Location);
    mean_distances = [mean_distances, mean(dists)];
    orientation_scores = [orientation_scores, abs(dot(axialvec, axvec))];
    cyl_models = [cyl_models, {model_new}];
    cyl_pcs = [cyl_pcs, {ptCloud_all}];

    % % plotting for visualization
    % ax = subplot(2, 2, i);
    % pcshowcus(ax, ptCloud_all, 'all', 'b');
    % hold(ax, 'on');
    % pcshowcus(ax, ptCloud, inlierIndices, 'g');
    % 
    % % plot fitted cylinder
    % m = plot(model, 'Color', 'red');
    % alpha(m, 0.1);
    % m = plot(model_new, 'Color', [0 0.7 0]);
    % alpha(m, 0.2);
    % 
    % % generate ocean surface plane
    % minx = min(points_all(:, 1)); maxx = max(points_all(:, 1));
    % miny = min(points_all(:, 2)); maxy = max(points_all(:, 2));
    % [x, y] = meshgrid(linspace(minx - 1, maxx + 1, 2), linspace(miny - 1, maxy + 1, 2));
    % z = zeros(size(x, 1));
    % s = surf(ax, x, y, z);
    % alpha(s, 0.05)
    % 
    % grid(ax, 'on')
    % box(ax, 'on')
    % axis(ax, 'equal')
end
