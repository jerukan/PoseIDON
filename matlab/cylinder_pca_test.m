%% Cylinder generation
% parameters
p1 = [-2, -2, -3];
p2 = [3, 2, 2]; 
R = 2;
% generation
N = 500; % # of points to generate
axialvec = p2 - p1;
axialvec = axialvec / norm(axialvec);
axialpoints = p1 + (p2 - p1) .* rand(N, 1);
% circr = sqrt(rand(N, 1)) * R;
circr = R + normrnd(0, 0.0, N, 1);
circtheta = rand(N, 1) * 2 * pi;
circpoints = [cos(circtheta) .* circr, sin(circtheta) .* circr];
axnull = null(axialvec);
points = axialpoints + circpoints * axnull.';

% Cylinder top/bottom generation
% for each end of the cylinder
N_cap = 60;
circr_cap = sqrt(rand(N_cap * 2, 1)) * R;
circtheta_cap = rand(N_cap * 2, 1) * 2 * pi;
circpoints_cap = [cos(circtheta_cap) .* circr_cap, sin(circtheta_cap) .* circr_cap];
% append to cylinder edge points
points = [points; p1 + circpoints_cap(1:N_cap, :) * axnull.'; p2 + circpoints_cap(N_cap:N_cap*2, :) * axnull.'];

% filter out buried points
points = points(points(:, 3) >= 0, :);

%% Fitting and stuff
ptCloud = pointCloud(points);

origin = mean(points);
coeffs = pca(points);

% ptCloud = pcread('models3d/barrelsingle-nonozzle.ply');

maxDistance = 0.5;
% half of the barrel
% roi = [-0.5 0.5; -0.5 0.5; 0 1];
% full barrel
% roi = [-0.5 0.5; -0.5 0.5; -1 1];
% everything
roi = [-inf inf; -inf inf; -inf inf];
sampleIndices = findPointsInROI(ptCloud, roi);

% referenceVector = [0, 0, 1];
cylmodels = cell(4, 1);
inlier_idxs = cell(4, 1);
cylerrs = zeros(4, 1);
cyldist_raw = zeros(4, 1);
for i=1:3
    refvec = coeffs(:, i);
    [model, inlierIndices, ~, meanError] = pcfitcylinder( ...
        ptCloud, ...
        maxDistance, ...
        refvec, ...
        SampleIndices=sampleIndices ...
    );
    cylmodels{i} = model;
    inlier_idxs{i} = inlierIndices;
    if isempty(meanError)
        cylerrs(i) = Inf;
        cyldist_raw(i) = Inf;
    else
        cylerrs(i) = meanError;
        cyldist_raw(i) = mean(evalCylinder(model.Parameters, ptCloud.Location));
    end
end
[model, inlierIndices, ~, meanError] = pcfitcylinder( ...
    ptCloud, ...
    maxDistance, ...
    SampleIndices=sampleIndices ...
);
cylmodels{4} = model;
inlier_idxs{4} = inlierIndices;
if isempty(meanError)
    cylerrs(4) = Inf;
    cyldist_raw(4) = Inf;
else
    cylerrs(4) = meanError;
    cyldist_raw(4) = mean(evalCylinder(model.Parameters, ptCloud.Location));
end

%% Plotting results
disp(cylerrs);
disp(cyldist_raw);

for i=1:4
    ax = subplot(1, 4, i);
    % plot3(points(:,1),points(:,2),points(:,3),'.')
    % plotpc(ptCloud, sampleIndices);
    pcshowcus(ax, ptCloud, 'all', 'b')
    hold(ax, 'on')
    pcshowcus(ax, ptCloud, inlier_idxs{i}, 'g')
    % pcshow(ptCloud);
    m = plot(cylmodels{i});
    alpha(m, 0.1);
    veclen = 2;
    if i < 4
        plotvec3(origin, coeffs(:,i), veclen, 'r-^', 3)
    end
    % plotvec3(origin, coeffs(:,2), veclen, 'g-^', 3)
    % plotvec3(origin, coeffs(:,3), veclen, 'b-^', 3)
    [x, y] = meshgrid(-5:2:5); % Generate x and y data
    z = zeros(size(x, 1)); % Generate z data
    s = surf(ax, x, y, z); % Plot the surface
    alpha(s, 0.05);
    grid(ax, 'on')
    box(ax, 'on')
    axis(ax, 'equal')
    if i < 4
        title(ax, ['PCA vec ', num2str(i)])
    else
        title(ax, 'no axis')
    end
    hold(ax, 'off')
end
