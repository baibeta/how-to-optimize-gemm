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

  size_t vl = vsetvlmax_e32m2(); // 8

  float 
    /* Point to the current elements in the four rows of A */
    *a_0p_pntr, *a_1p_pntr, *a_2p_pntr, *a_3p_pntr, *a_4p_pntr, *a_5p_pntr, *a_6p_pntr, *a_7p_pntr;


  a_0p_pntr = &A(0, 0);
  a_1p_pntr = &A(1, 0);
  a_2p_pntr = &A(2, 0);
  a_3p_pntr = &A(3, 0);
  a_4p_pntr = &A(4, 0);
  a_5p_pntr = &A(5, 0);
  a_6p_pntr = &A(6, 0);
  a_7p_pntr = &A(7, 0);


  vfloat32m2_t c_p0_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p1_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p2_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p3_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p4_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p5_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p6_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p7_sum = vfmv_v_f_f32m2(0.0f, vl);

  float
    a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg,
    a_4p_reg, a_5p_reg, a_6p_reg, a_7p_reg;

  for (int p = 0; p < k; ++p) {
    vfloat32m2_t  b_reg = vle32_v_f32m2 (&B(p, 0), vl);

    a_0p_reg = *a_0p_pntr++;
    a_1p_reg = *a_1p_pntr++;
    a_2p_reg = *a_2p_pntr++;
    a_3p_reg = *a_3p_pntr++;
    a_4p_reg = *a_4p_pntr++;
    a_5p_reg = *a_5p_pntr++;
    a_6p_reg = *a_6p_pntr++;
    a_7p_reg = *a_7p_pntr++;

    c_p0_sum = vfmacc_vf_f32m2(c_p0_sum, a_0p_reg,  b_reg, vl);
    c_p1_sum = vfmacc_vf_f32m2(c_p1_sum, a_1p_reg,  b_reg, vl);
    c_p2_sum = vfmacc_vf_f32m2(c_p2_sum, a_2p_reg,  b_reg, vl);
    c_p3_sum = vfmacc_vf_f32m2(c_p3_sum, a_3p_reg,  b_reg, vl);
    c_p4_sum = vfmacc_vf_f32m2(c_p4_sum, a_4p_reg,  b_reg, vl);
    c_p5_sum = vfmacc_vf_f32m2(c_p5_sum, a_5p_reg,  b_reg, vl);
    c_p6_sum = vfmacc_vf_f32m2(c_p6_sum, a_6p_reg,  b_reg, vl);
    c_p7_sum = vfmacc_vf_f32m2(c_p7_sum, a_7p_reg,  b_reg, vl);
  }

    vse32_v_f32m2(&C(0, 0), c_p0_sum, vl);
    vse32_v_f32m2(&C(1, 0), c_p1_sum, vl);
    vse32_v_f32m2(&C(2, 0), c_p2_sum, vl);
    vse32_v_f32m2(&C(3, 0), c_p3_sum, vl);
    vse32_v_f32m2(&C(4, 0), c_p4_sum, vl);
    vse32_v_f32m2(&C(5, 0), c_p5_sum, vl);
    vse32_v_f32m2(&C(6, 0), c_p6_sum, vl);
    vse32_v_f32m2(&C(7, 0), c_p7_sum, vl);
}


void MY_MMult_8x8_m2( int m, int n, int k, float *a, int lda, 
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
