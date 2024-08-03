function volume = cylVolume(model)
    volume = model.Height * model.Parameters(7)^2 * pi;
end

