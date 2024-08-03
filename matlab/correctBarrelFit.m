function corrModel = correctBarrelFit(model, r, h)
% Attempts to correct dimensions of a fitted barrel by scaling it while
% aligning it with the highest point above the surface of the cylinder
    p1_0 = model.Parameters(1:3);
    p2_0 = model.Parameters(4:6);
    r0 = model.Parameters(7);
    % force axial vector to point downwards
    if p1_0(3) > p2_0(3)
        above = p1_0;
        axvec = p2_0 - above;
    else
        above = p2_0;
        axvec = p1_0 - above;
    end
    axvec = axvec / norm(axvec);
    hor_len = sqrt(axvec(1)^2 + axvec(2)^2);
    if hor_len < 0.001
        belownew = above + axvec * h;
        corrModel = cylinderModel([belownew, above, r]);
    else
        % should still be normal
        % this is normal to axis vector, and points up (+z)
        axvec_normup = [
            abs(axvec(3)) * axvec(1) / hor_len, ...
            abs(axvec(3)) * axvec(2) / hor_len, ...
            hor_len
        ];
        abovenew = (r0 - r) * axvec_normup + above;
        belownew = abovenew + axvec * h;
        corrModel = cylinderModel([belownew, abovenew, r]);
    end
end
