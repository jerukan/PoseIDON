function plotvec3(origin, vec, length, linesty, lwidth)
origin = reshape(origin, [], 1);
vec = reshape(vec, [], 1);
normalized = (vec / norm(vec)) * length;
normalized = origin + normalized;
plot3([origin(1) normalized(1)], [origin(2) normalized(2)], [origin(3) normalized(3)], linesty, 'LineWidth', lwidth);
end
