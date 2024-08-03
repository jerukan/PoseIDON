% hardcoded for the output of one of the barrel fitting trials
% calculates and stores results of the convex hull for above surface
% point clouds
chulls = cell(1,913);
vols = zeros(913, 1);
for i = 1:913
    xyzs = cyl_pcs_above{i}.Location;
    if size(xyzs, 1) < 5
        continue
    end
    [k, av] = convhull(cyl_pcs_above{i}.Location);
    chulls{i} = k;
    vols(i) = av;
end
