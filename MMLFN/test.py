### Import


###

parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--polar', type=bool, default=False)
parser.add_argument('--number_of_epoch', type=int, default=40) #not needed of course but needed in the trainer init
parser.add_argument('--augmentations', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--FOV', type=int, help='70, 90, 180, 360', default=360)
parser.add_argument('--model_type', type=str, help="dual, triple_sat, triple_grd, quadruple, quintuple", default="quadruple")
parser.add_argument('--quantization', type=str, help="16-true, 16-mixed", default="16-true")

args = parser.parse_args()

use_polar = args.polar
number_of_epoch = args.number_of_epoch
augmentations = args.augmentations
batch_size = args.batch_size
model_FOV = args.FOV
model_type = args.model_type
quantization = args.quantization

loss_function = CombinedLoss(loss_function = nn.CrossEntropyLoss(label_smoothing = 0.1))


###

def test():

    # @title Creating dataloaders

    input_dir = '/content/input/data/CVUSA'

    data_module = CVUSADataModule(
        input_dir = input_dir,
        polar = use_polar,
        augmentations = augmentations,
        batch_size = batch_size,
        # grd_resize = 128,
        # grd_seg_resize = 128,
        # grd_dep_resize = 128,
        # sat_resize = 128, #256
        # sat_seg_resize = 128, #256
        fov = model_FOV,
        all_rgb = False
    )

    data_module.setup()

    #train_loader = data_module.train_dataloader() not needed
    val_loader = data_module.val_dataloader()

    ###

    if model_type == "dual":

        grd_model = ConvNeXtBranch(output_dim = 768, use_embeddings = True, model_type='Tiny')
        sat_model = ConvNeXtBranch(output_dim = 768, use_embeddings = True, model_type='Tiny')

        model = DualModel(
            model_grd = grd_model,
            model_sat = sat_model,
            loss = loss_function
        )


    elif model_type == "triple_sat":

        grd_model = ConvNeXtBranch(output_dim = 768, use_embeddings = True, model_type='Tiny')
        seg_model = ConvNeXtBranch(output_dim = 256, use_embeddings = True, model_type='Tiny')
        sat_model = ConvNeXtBranch(output_dim = 512, use_embeddings = True, model_type='Tiny')

        model = TripleModel_sat(
            model_grd = grd_model,
            model_sat = sat_model,
            model_sat_seg = seg_model,
            loss = loss_function
        )


    elif model_type == "triple_grd":

        grd_model = ConvNeXtBranch(output_dim = 512, use_embeddings = True, model_type='Tiny')
        seg_model = ConvNeXtBranch(output_dim = 256, use_embeddings = True, model_type='Tiny')
        sat_model = ConvNeXtBranch(output_dim = 768, use_embeddings = True, model_type='Tiny')

        model = TripleModel_grd(
            model_grd = grd_model,
            model_sat = sat_model,
            model_grd_seg = seg_model,
            loss = loss_function
        )


    elif model_type == "quadruple":

        grd_model = ConvNeXtBranch(output_dim = 756, use_embeddings = True, model_type='Tiny')
        grd_seg_model = ConvNeXtBranch(output_dim = 256, use_embeddings = True, model_type='Tiny')
        sat_model = ConvNeXtBranch(output_dim = 756, use_embeddings = True, model_type='Tiny')
        sat_seg_model = ConvNeXtBranch(output_dim = 256, use_embeddings = True, model_type='Tiny')

        model = QuadrupleModel(
            model_grd = grd_model,
            model_grd_seg = grd_seg_model,
            model_sat = sat_model,
            model_sat_seg = sat_seg_model,
            loss = loss_function
        )


    elif model_type == "quintuple":

        grd_model = ConvNeXtBranch(output_dim = 512, use_embeddings = True, model_type='Tiny')
        grd_seg_model = ConvNeXtBranch(output_dim = 192, use_embeddings = True, model_type='Tiny')
        grd_dep_model = ConvNeXtBranch(output_dim = 64, use_embeddings = True, model_type='Tiny')
        sat_model = ConvNeXtBranch(output_dim = 512, use_embeddings = True, model_type='Tiny')
        sat_seg_model = ConvNeXtBranch(output_dim = 256, use_embeddings = True, model_type='Tiny')

        model = QuintupleModel(
            model_grd = grd_model,
            model_grd_seg = grd_seg_model,
            model_grd_dep = grd_dep_model,
            model_sat = sat_model,
            model_sat_seg = sat_seg_model,
            loss = loss_function
        )


    ###

    # @title Create Trainer
    progress_bar = RichProgressBar()

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=number_of_epoch,
        devices=1, # CHECK
        callbacks=[progress_bar],
        accumulate_grad_batches=4,
        gradient_clip_val=100,
        precision = quantization
    )
    ###

    # @title Test
    
    trainer.test(
        model = model,
        dataloaders = val_loader,
        ckpt_path = 'path to use'
    )

    ###


if __name__ == '__main__':
    test()



