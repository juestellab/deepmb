function shearlet_l1_functional = get_shearlet_L1_functional(img_size)
shearletSystem = SLgetShearletSystem2D(0, img_size(1), img_size(2), 4);
shearletTrafo = @(x) SLsheardec2D(reshape(x, img_size(1), img_size(2)), shearletSystem);
shearlet_l1_functional = @(x) PhiForShearletReg_noparfor(reshape(x, img_size(1), img_size(2), []), shearletTrafo);
end

