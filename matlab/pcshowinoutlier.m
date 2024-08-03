% show RANSAC inliers/outliers of the result of shape fitting
function pcshowinoutlier(pc, inlieridxs, outlieridxs)
pcshow(pointCloud(select(pc, inlieridxs).Location, Color='g'))
hold on
pcshow(pointCloud(select(pc, outlieridxs).Location, Color='r'))
hold off
end
