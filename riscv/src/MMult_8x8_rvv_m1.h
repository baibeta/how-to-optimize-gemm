#include <stdio.h>
#include <riscv_vector.h>

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]


/* Routine for computing C = A * B + C */

void AddDot8x8( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  /* So, this routine computes a 4x8 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  
	   C( 4, 0 )...
	   C( 5, 0 )...
	   C( 6, 0 )...
           C( 7, 0 )...

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	   C( i+4, j )...
           C( i+5, j )...
	   C( i+6, j )...
           C( i+7, j )...
	  
     in the original matrix C 

     In this version, we use registers for elements in the current row
     of B as well */

  size_t vl = vsetvlmax_e32m1();

  vfloat32m1_t c_p0_sum = vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c_p1_sum = vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c_p2_sum = vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c_p3_sum = vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c_p4_sum = vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c_p5_sum = vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c_p6_sum = vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c_p7_sum = vfmv_v_f_f32m1(0.0f, vl);

  vfloat32m1_t c_p04_sum = vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c_p14_sum = vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c_p24_sum = vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c_p34_sum = vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c_p44_sum = vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c_p54_sum = vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c_p64_sum = vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t c_p74_sum = vfmv_v_f_f32m1(0.0f, vl);

  for (int p = 0; p < k; ++p) {
    vfloat32m1_t  b_reg = vle32_v_f32m1 (&B(p, 0), vl);
    vfloat32m1_t  b_reg4 = vle32_v_f32m1 (&B(p, 4), vl);

    c_p0_sum = vfmacc_vf_f32m1(c_p0_sum, A(0, p),  b_reg, vl);
    c_p1_sum = vfmacc_vf_f32m1(c_p1_sum, A(1, p),  b_reg, vl);
    c_p2_sum = vfmacc_vf_f32m1(c_p2_sum, A(2, p),  b_reg, vl);
    c_p3_sum = vfmacc_vf_f32m1(c_p3_sum, A(3, p),  b_reg, vl);
    c_p4_sum = vfmacc_vf_f32m1(c_p4_sum, A(4, p),  b_reg, vl);
    c_p5_sum = vfmacc_vf_f32m1(c_p5_sum, A(5, p),  b_reg, vl);
    c_p6_sum = vfmacc_vf_f32m1(c_p6_sum, A(6, p),  b_reg, vl);
    c_p7_sum = vfmacc_vf_f32m1(c_p7_sum, A(7, p),  b_reg, vl);

    c_p04_sum = vfmacc_vf_f32m1(c_p04_sum, A(0, p),  b_reg4, vl);
    c_p14_sum = vfmacc_vf_f32m1(c_p14_sum, A(1, p),  b_reg4, vl);
    c_p24_sum = vfmacc_vf_f32m1(c_p24_sum, A(2, p),  b_reg4, vl);
    c_p34_sum = vfmacc_vf_f32m1(c_p34_sum, A(3, p),  b_reg4, vl);
    c_p44_sum = vfmacc_vf_f32m1(c_p44_sum, A(4, p),  b_reg4, vl);
    c_p54_sum = vfmacc_vf_f32m1(c_p54_sum, A(5, p),  b_reg4, vl);
    c_p64_sum = vfmacc_vf_f32m1(c_p64_sum, A(6, p),  b_reg4, vl);
    c_p74_sum = vfmacc_vf_f32m1(c_p74_sum, A(7, p),  b_reg4, vl);
  }

    vse32_v_f32m1(&C(0, 0), c_p0_sum, vl);
    vse32_v_f32m1(&C(1, 0), c_p1_sum, vl);
    vse32_v_f32m1(&C(2, 0), c_p2_sum, vl);
    vse32_v_f32m1(&C(3, 0), c_p3_sum, vl);
    vse32_v_f32m1(&C(4, 0), c_p4_sum, vl);
    vse32_v_f32m1(&C(5, 0), c_p5_sum, vl);
    vse32_v_f32m1(&C(6, 0), c_p6_sum, vl);
    vse32_v_f32m1(&C(7, 0), c_p7_sum, vl);

    vse32_v_f32m1(&C(0, 4), c_p04_sum, vl);
    vse32_v_f32m1(&C(1, 4), c_p14_sum, vl);
    vse32_v_f32m1(&C(2, 4), c_p24_sum, vl);
    vse32_v_f32m1(&C(3, 4), c_p34_sum, vl);
    vse32_v_f32m1(&C(4, 4), c_p44_sum, vl);
    vse32_v_f32m1(&C(5, 4), c_p54_sum, vl);
    vse32_v_f32m1(&C(6, 4), c_p64_sum, vl);
    vse32_v_f32m1(&C(7, 4), c_p74_sum, vl);
}


void MY_MMult_8x8_m1( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=8 ){
    for ( i=0; i<m; i+=8 ){        /* Loop over the rows of C */
      AddDot8x8( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
    }
  }
}
