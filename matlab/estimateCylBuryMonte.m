function burialRatio = estimateCylBuryMonte(model, npoints, axidx)
    % monte carlo estimation of cylinder burial ratio
    xyzs = randCylPts(model, npoints);
    undermask = xyzs(:, axidx) <= 0;
    burialRatio = sum(undermask) / npoints;
end
