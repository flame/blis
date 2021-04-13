#define SVE512_IN_REG_TRANSPOSE_d8x2(DST0,DST1,DST2,DST3,DST4,DST5,DST6SRC0,DST7SRC1,PT,P2C,P4C,P6C) \
  "trn1    " #DST0".d, " #DST6SRC0".d, " #DST7SRC1".d \n\t" \
  "trn2    " #DST1".d, " #DST6SRC0".d, " #DST7SRC1".d \n\t" \
  "compact " #DST2".d, " #P2C", " #DST0".d \n\t" \
  "compact " #DST3".d, " #P2C", " #DST1".d \n\t" \
  "compact " #DST4".d, " #P4C", " #DST0".d \n\t" \
  "compact " #DST5".d, " #P4C", " #DST1".d \n\t" \
  "compact " #DST6SRC0".d, " #P6C", " #DST0".d \n\t" \
  "compact " #DST7SRC1".d, " #P6C", " #DST1".d \n\t"

