function plotpcnorms(pc, skip)
    normals = pc.Normal;
    x = pc.Location(1:skip:end,1);
    y = pc.Location(1:skip:end,2);
    z = pc.Location(1:skip:end,3);
    u = normals(1:skip:end,1);
    v = normals(1:skip:end,2);
    w = normals(1:skip:end,3);
    quiver3(x,y,z,u,v,w);
end
