function xyzs = randSpherePtsSurf(model, npoints, sigma)
    % generates uniformly distributed points on the surface of a sphere
    c = model.Parameters(1:3);
    r = model.Parameters(4);
    
    % apparently the standard multivariate normal distribution
    % is rotation invariant, so its distributed uniformly
    unnormxyzs = normrnd(0, 1, npoints, 3);
    xyzs = unnormxyzs ./ vecnorm(unnormxyzs, 2, 2);
    xyzs = c + xyzs .* r;
    xyzs = xyzs + normrnd(0, sigma, size(xyzs));
end
