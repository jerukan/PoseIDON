% plotting for visualization
i = 7;
ax = subplot(1, 1, 1);
pcshow(pointCloud(cyl_pcs{i}.Location, Color='r'))
% pcshowcus(ax, cyl_pcs{i}, 'all', 'b');
hold(ax, 'on');
inliers = pointCloud(select(cyl_pcs_above{i}, cyl_pcs_inliersidxs{i}).Location, Color='g');
inliers.Normal = pcnormals(inliers, 50);
pcshow(inliers)
plotpcnorms(inliers, 10);
% pcshowcus(ax, cyl_pcs_above{i}, inlierIndices, 'g');

% plot fitted cylinder
% m = plot(cyl_truth{i}, 'Color', 'red');
% alpha(m, 0.1);
m = plot(cyl_models{i}, 'Color', [0 0.7 0]);
alpha(m, 0.2);
m = plot(cyl_uncorrected{i}, 'Color', [0.5 0.8 0]);
alpha(m, 0.2);


% generate ocean surface plane
minx = cyl_pcs{i}.XLimits(1); maxx = cyl_pcs{i}.XLimits(2);
miny = cyl_pcs{i}.YLimits(1); maxy = cyl_pcs{i}.YLimits(2);
[x, y] = meshgrid(linspace(minx - 1, maxx + 1, 2), linspace(miny - 1, maxy + 1, 2));
z = zeros(size(x, 1));
s = surf(ax, x, y, z);
alpha(s, 0.2)

grid(ax, 'on')
box(ax, 'on')
axis(ax, 'equal')
hold(ax, 'off')
