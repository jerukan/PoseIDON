function xyzs = randCylPts(model, npoints)
    % generates uniformly distributed points within the volume of
    % a defined cylinder
    % returns nx3 array
    axis = model.Orientation;
    axis = axis / norm(axis);
    axnull = null(axis);
    axnull1 = axnull(:, 1);
    axnull2 = axnull(:, 2);
    x1 = model.Parameters(1:3);
    r = model.Radius;
    h = model.Height;
    rand_r = r * sqrt(rand(1, npoints));
    rand_theta = rand(1, npoints) * 2 * pi;
    rand_h = rand(1, npoints) * h;
    
    cosval = repmat(axnull1, 1, npoints) .* rand_r .* cos(rand_theta);
    sinval = repmat(axnull2, 1, npoints) .* rand_r .* sin(rand_theta);
    xyzs = cosval + sinval + repmat(x1', 1, npoints) + rand_h .* repmat(axis', 1, npoints);
    xyzs = xyzs';
end
