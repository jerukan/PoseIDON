barrelxyz = rotscenexyz(isbarrel, :);
barrelpc = pointCloud(barrelxyz);
barrelpc.Normal = pcnormals(barrelpc, 15);
[barrelcyl, barrelinidxs, barreloutidxs] = pcfitcylinder(barrelpc, 0.1);
estimatedBurial = estimateCylBury(barrelcyl, 1000) / cylVolume(barrelcyl);

normals = barrelpc.Normal;
skip = 10;
x = barrelpc.Location(1:skip:end,1);
y = barrelpc.Location(1:skip:end,2);
z = barrelpc.Location(1:skip:end,3);
u = normals(1:skip:end,1);
v = normals(1:skip:end,2);
w = normals(1:skip:end,3);

% fitted cylinder on whole scene
pcshow(rotscenepc)
hold on
b = plot(barrelcyl);
alpha(b, 0.3)
% generate ocean surface plane
minx = min(rotscenexyz(:, 1)); maxx = max(rotscenexyz(:, 1));
miny = min(rotscenexyz(:, 2)); maxy = max(rotscenexyz(:, 2));
[x, y] = meshgrid(linspace(minx - 1, maxx + 1, 2), linspace(miny - 1, maxy + 1, 2));
z = zeros(size(x, 1));
s = surf(x, y, z);
alpha(s, 0.2)
hold off