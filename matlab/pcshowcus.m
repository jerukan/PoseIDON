function pcshowcus(ax, pc, idxs, colr)
if strcmp(idxs, 'all')
    plot3(ax, pc.Location(:, 1), pc.Location(:, 2), pc.Location(:, 3), [colr, '.'])
else
    plot3(ax, pc.Location(idxs, 1), pc.Location(idxs, 2), pc.Location(idxs, 3), strcat(colr,'.'))
end
end
