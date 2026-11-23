S = '/data/alpamayo-bench/lab/src/'
SWAPS = {'vision_attention_max': (S+'vattn_direct_nowpe.cpp', S+'vattn_direct_nowpe.hsaco'),
         'vision_attention': (S+'vattn_plain_direct.cpp', S+'vattn_plain_direct.hsaco'),
         'vision_fc1_gelu_max': (S+'fc1_nosmall.cpp', S+'fc1_nosmall.hsaco')}
exec(open(S+'refreeze_objects.py').read())
import os
print(sorted(os.listdir(OUT)), sorted(os.listdir(OUT/'gfx1200')))
