.text
.align 5

.globl TEST, TEST_ADD, TEST_S

TEST:
    vsetvli t0, a0, e32, m1  	# 设置向量寄存器的长度和类型为 32 位浮点数
    mv      t1, a0		# 将 a0 的值（循环次数）存储到 t1

.loop2:
    vfmacc.vv v1, v1, v2
    vfmacc.vv v3, v3, v4
    vfmacc.vv v5, v5, v6
    vfmacc.vv v7, v7, v8
    vfmacc.vv v9, v9, v10       
    vfmacc.vv v11, v11, v12    
    vfmacc.vv v13, v13, v14   
    vfmacc.vv v15, v15, v16  
    vfmacc.vv v17, v17, v18
    vfmacc.vv v19, v19, v20


    vfmacc.vv v1, v1, v2
    vfmacc.vv v3, v3, v4
    vfmacc.vv v5, v5, v6
    vfmacc.vv v7, v7, v8
    vfmacc.vv v9, v9, v10
    vfmacc.vv v11, v11, v12
    vfmacc.vv v13, v13, v14
    vfmacc.vv v15, v15, v16
    vfmacc.vv v17, v17, v18
    vfmacc.vv v19, v19, v20

    vfmacc.vv v1, v1, v2
    vfmacc.vv v3, v3, v4
    vfmacc.vv v5, v5, v6
    vfmacc.vv v7, v7, v8
    vfmacc.vv v9, v9, v10
    vfmacc.vv v11, v11, v12
    vfmacc.vv v13, v13, v14
    vfmacc.vv v15, v15, v16
    vfmacc.vv v17, v17, v18
    vfmacc.vv v19, v19, v20


    addi t1, t1, -1


    bnez t1, .loop2             # 如果 t1 != 0，跳转到 .loop2

    ret                         # 返回


TEST_ADD:
    vsetvli t0, a0, e32, m1     # 设置向量寄存器的长度和类型为 32 位浮点数
    mv      t1, a0              # 将 a0 的值（循环次数）存储到 t1

.loop3:
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1

    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1

    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1
    vfmacc.vv v1, v1, v1



    addi t1, t1, -1


    bnez t1, .loop3             # 如果 t1 != 0，跳转到 .loop2

    ret                         # 返回




TEST_S:
	mv	t1, a0
.loop4:
        addi    t2,t2,2
	addi	t3,t3,3
	addi	t4,t4,4
	addi	t5,t5,5
	addi	t6,t6,6

	addi    t2,t2,2
        addi    t3,t3,3
        addi    t4,t4,4
        addi    t5,t5,5
        addi    t6,t6,6

	addi    t2,t2,2
        addi    t3,t3,3
        addi    t4,t4,4
        addi    t5,t5,5
        addi    t6,t6,6

	addi    t2,t2,2
        addi    t3,t3,3
        addi    t4,t4,4
        addi    t5,t5,5
        addi    t6,t6,6

	addi    t2,t2,2
        addi    t3,t3,3
        addi    t4,t4,4
        addi    t5,t5,5
        addi    t6,t6,6

	addi    t2,t2,2
        addi    t3,t3,3
        addi    t4,t4,4
        addi    t5,t5,5
        addi    t6,t6,6
	
	addi	t1,t1,-1
	bnez t1, .loop4
ret
