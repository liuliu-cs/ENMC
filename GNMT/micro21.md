# Python Script
python3 translate.py --input data/wmt16_de_en/newstest2014.en --reference data/wmt16_de_en/newstest2014.de --output /tmp/output --model gnmt-dre-e1/model_best.pth

## In `translate.py`
        model.decoder.dre_init()
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    model.decoder.dre_activate(enable_dre=True, dre_inference=True, profiling=True, 
        qtize=4, num_candidates=256)