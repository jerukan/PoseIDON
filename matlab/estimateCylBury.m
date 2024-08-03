function burialVolume = estimateCylBury(model, steps)
    % basically a Riemann approximation of the cylinder volume
    % this estimator will always underestimate burial volume
    % the amount it underestimates by will decrease with more steps
    % calculations are simplified by rotating the cylinder so that its
    % axis vector is parallel to the x-axis.
    x1 = model.Parameters(1:3).';
    r = model.Parameters(7);
    axvec = model.Orientation.';
    axvec = axvec / norm(axvec);
    len2d = sqrt(axvec(1)^2 + axvec(2)^2);
    % the normal vector to the axis vector that points up in the
    % z-axis
    % note this basically rotates the cylinder to be aligned with
    % the y-axis to make calculations easier. relative orientation
    % to seafloor is unchanged.
    ax_normvec = [-axvec(3); 0; len2d];
    ax_normvec = ax_normvec / norm(ax_normvec);
    
    stepsize = model.Height / steps;

    burialVolume = 0;
    for i = 1:steps
        axpoint = x1 + axvec * stepsize * i;
        axpt_to_ground = ax_normvec * (-axpoint(3) / ax_normvec(3));
        % if the cylinder is completely vertical, this will be NaN and
        % false, so it'll be fine
        if norm(axpt_to_ground) < r
            % circle intersects ground, calculate circular segment info
            h = r - norm(axpt_to_ground);
            % circular segment area
            % formula from wikipedia
            a = r^2 * acos(1 - (h / r)) - (r - h) * sqrt(r^2 - (r - h)^2);
            if axpt_to_ground(3) > 0
                % the point on axis is buried, take complement of circular
                % segment area
                a = pi * r^2 - a;
            end
            stepvol = a * stepsize;
        else
            % circle is completely underground or above ground
            if axpoint(3) < 0
                stepvol = pi * r^2 * stepsize;
            else
                stepvol = 0;
            end
        end
        burialVolume = burialVolume + stepvol;
    end
end
