# TFPG
Test For PaddlePaddle GPU/A100

#### 1、Big file README
* #1
```sql
# find paddle1/* -type f -size +1M | xargs ls -Slh
-rwxr-xr-x@ 1 ----  staff   1.9G 11 20 16:33 paddle1/fluid/libpaddle.so
-rwxr-xr-x@ 1 ----  staff   124M 11 20 16:33 paddle1/libs/libmklml_intel.so
-rwxr-xr-x@ 1 ----  staff    35M 11 20 16:33 paddle1/libs/libmkldnn.so.0
-rwxr-xr-x@ 1 ----  staff    35M 11 20 16:33 paddle1/libs/libdnnl.so.1
-rwxr-xr-x@ 1 ----  staff    35M 11 20 16:33 paddle1/libs/libdnnl.so.2
-rwxr-xr-x@ 1 ----  staff    12M 11 20 16:33 paddle1/libs/libwarpctc.so
-rwxr-xr-x@ 1 ----  staff   9.1M 11 20 16:33 paddle1/libs/liblapack.so.3
-rwxr-xr-x@ 1 ----  staff   1.8M 11 20 16:33 paddle1/libs/libiomp5.so
-rwxr-xr-x@ 1 ----  staff   1.2M 11 20 16:33 paddle1/libs/libgfortran.so.3
```

* #2
```sql
# find paddle2/* -type f -size +1M | xargs ls -Slh
-rwxr-xr-x@ 1 ----  staff   718M  2 17 11:17 paddle2/fluid/core_avx.so
-rwxr-xr-x@ 1 ----  staff   128M  2 17 11:17 paddle2/libs/libmklml_intel.so
-rw-r--r--@ 1 ----  staff    35M  2 17 11:17 paddle2/libs/libmkldnn.so.0
-rw-r--r--@ 1 ----  staff    35M  2 17 11:17 paddle2/libs/libdnnl.so.1
-rw-r--r--@ 1 ----  staff    35M  2 17 11:17 paddle2/libs/libdnnl.so.2
-rw-r--r--@ 1 ----  staff    12M  2 17 11:17 paddle2/libs/libwarpctc.so
-rwxr-xr-x@ 1 ----  staff   9.1M  2 17 11:17 paddle2/libs/liblapack.so.3
-rwxr-xr-x@ 1 ----  staff   2.1M  2 17 11:17 paddle2/libs/libiomp5.so
-rwxr-xr-x@ 1 ----  staff   1.2M  2 17 11:17 paddle2/libs/libgfortran.so.3
```

#### 2、#1 diff #2
> https://github.com/IvanaXu/TestPaddleA100/commit/1ae289bfb0baaaaa5d2cd7c9d8d893f61b71b01f
