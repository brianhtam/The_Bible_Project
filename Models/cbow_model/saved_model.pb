�
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.12v2.3.0-54-gfcc4b966f18��
�
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�&d*'
shared_nameembedding_1/embeddings
�
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes
:	�&d*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�&*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	d�&*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�&*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�&*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
b


embeddings
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
 


0
1
2
 


0
1
2
�
metrics

layers
trainable_variables
layer_metrics
non_trainable_variables
layer_regularization_losses
regularization_losses
	variables
 
fd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE


0
 


0
�
metrics

layers
trainable_variables
 layer_metrics
!non_trainable_variables
"layer_regularization_losses
regularization_losses
	variables
 
 
 
�
#metrics

$layers
trainable_variables
%layer_metrics
&non_trainable_variables
'layer_regularization_losses
regularization_losses
	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
(metrics

)layers
trainable_variables
*layer_metrics
+non_trainable_variables
,layer_regularization_losses
regularization_losses
	variables
 

0
1
2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
!serving_default_embedding_1_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall!serving_default_embedding_1_inputembedding_1/embeddingsdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������&*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_14631049
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_1/embeddings/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_14631200
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_1/embeddingsdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_14631219��
�
�
J__inference_sequential_1_layer_call_and_return_conditional_losses_14631068

inputs)
%embedding_1_embedding_lookup_14631053*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity�u
embedding_1/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding_1/Cast�
embedding_1/embedding_lookupResourceGather%embedding_1_embedding_lookup_14631053embedding_1/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_1/embedding_lookup/14631053*+
_output_shapes
:���������d*
dtype02
embedding_1/embedding_lookup�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_1/embedding_lookup/14631053*+
_output_shapes
:���������d2'
%embedding_1/embedding_lookup/Identity�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d2)
'embedding_1/embedding_lookup/Identity_1�
lambda_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2!
lambda_1/Mean/reduction_indices�
lambda_1/MeanMean0embedding_1/embedding_lookup/Identity_1:output:0(lambda_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������d2
lambda_1/Mean�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d�&*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMullambda_1/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������&2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�&*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������&2
dense_1/BiasAddz
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������&2
dense_1/Softmaxn
IdentityIdentitydense_1/Softmax:softmax:0*
T0*(
_output_shapes
:����������&2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������::::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_14631132

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:���������d2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
I__inference_embedding_1_layer_call_and_return_conditional_losses_14630909

inputs
embedding_lookup_14630903
identity�]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_14630903Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/14630903*+
_output_shapes
:���������d*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/14630903*+
_output_shapes
:���������d2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d2
embedding_lookup/Identity_1|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������d2

Identity"
identityIdentity:output:0**
_input_shapes
:���������::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_14631049
embedding_1_input
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������&*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_146308952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������&2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:���������
+
_user_specified_nameembedding_1_input
�
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_14631138

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:���������d2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
J__inference_sequential_1_layer_call_and_return_conditional_losses_14631027

inputs
embedding_1_14631017
dense_1_14631021
dense_1_14631023
identity��dense_1/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_14631017*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_embedding_1_layer_call_and_return_conditional_losses_146309092%
#embedding_1/StatefulPartitionedCall�
lambda_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_146309332
lambda_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_1_14631021dense_1_14631023*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_146309572!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*
T0*(
_output_shapes
:����������&2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference__wrapped_model_14630895
embedding_1_input6
2sequential_1_embedding_1_embedding_lookup_146308807
3sequential_1_dense_1_matmul_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resource
identity��
sequential_1/embedding_1/CastCastembedding_1_input*

DstT0*

SrcT0*'
_output_shapes
:���������2
sequential_1/embedding_1/Cast�
)sequential_1/embedding_1/embedding_lookupResourceGather2sequential_1_embedding_1_embedding_lookup_14630880!sequential_1/embedding_1/Cast:y:0*
Tindices0*E
_class;
97loc:@sequential_1/embedding_1/embedding_lookup/14630880*+
_output_shapes
:���������d*
dtype02+
)sequential_1/embedding_1/embedding_lookup�
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0*
T0*E
_class;
97loc:@sequential_1/embedding_1/embedding_lookup/14630880*+
_output_shapes
:���������d24
2sequential_1/embedding_1/embedding_lookup/Identity�
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d26
4sequential_1/embedding_1/embedding_lookup/Identity_1�
,sequential_1/lambda_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_1/lambda_1/Mean/reduction_indices�
sequential_1/lambda_1/MeanMean=sequential_1/embedding_1/embedding_lookup/Identity_1:output:05sequential_1/lambda_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������d2
sequential_1/lambda_1/Mean�
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	d�&*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOp�
sequential_1/dense_1/MatMulMatMul#sequential_1/lambda_1/Mean:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������&2
sequential_1/dense_1/MatMul�
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�&*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOp�
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������&2
sequential_1/dense_1/BiasAdd�
sequential_1/dense_1/SoftmaxSoftmax%sequential_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������&2
sequential_1/dense_1/Softmax{
IdentityIdentity&sequential_1/dense_1/Softmax:softmax:0*
T0*(
_output_shapes
:����������&2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������::::Z V
'
_output_shapes
:���������
+
_user_specified_nameembedding_1_input
�
�
E__inference_dense_1_layer_call_and_return_conditional_losses_14631159

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�&*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������&2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�&*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������&2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:����������&2	
Softmaxf
IdentityIdentitySoftmax:softmax:0*
T0*(
_output_shapes
:����������&2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
J__inference_sequential_1_layer_call_and_return_conditional_losses_14630974
embedding_1_input
embedding_1_14630918
dense_1_14630968
dense_1_14630970
identity��dense_1/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputembedding_1_14630918*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_embedding_1_layer_call_and_return_conditional_losses_146309092%
#embedding_1/StatefulPartitionedCall�
lambda_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_146309272
lambda_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_1_14630968dense_1_14630970*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_146309572!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*
T0*(
_output_shapes
:����������&2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:Z V
'
_output_shapes
:���������
+
_user_specified_nameembedding_1_input
�
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_14630927

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:���������d2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
$__inference__traced_restore_14631219
file_prefix+
'assignvariableop_embedding_1_embeddings%
!assignvariableop_1_dense_1_kernel#
assignvariableop_2_dense_1_bias

identity_4��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp'assignvariableop_embedding_1_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_1_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_1_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_3Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_3�

Identity_4IdentityIdentity_3:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2*
T0*
_output_shapes
: 2

Identity_4"!

identity_4Identity_4:output:0*!
_input_shapes
: :::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
J__inference_sequential_1_layer_call_and_return_conditional_losses_14630987
embedding_1_input
embedding_1_14630977
dense_1_14630981
dense_1_14630983
identity��dense_1/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputembedding_1_14630977*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_embedding_1_layer_call_and_return_conditional_losses_146309092%
#embedding_1/StatefulPartitionedCall�
lambda_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_146309332
lambda_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_1_14630981dense_1_14630983*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_146309572!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*
T0*(
_output_shapes
:����������&2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:Z V
'
_output_shapes
:���������
+
_user_specified_nameembedding_1_input
�
G
+__inference_lambda_1_layer_call_fn_14631143

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_146309272
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
t
.__inference_embedding_1_layer_call_fn_14631126

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_embedding_1_layer_call_and_return_conditional_losses_146309092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������d2

Identity"
identityIdentity:output:0**
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_1_layer_call_and_return_conditional_losses_14630957

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�&*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������&2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�&*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������&2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:����������&2	
Softmaxf
IdentityIdentitySoftmax:softmax:0*
T0*(
_output_shapes
:����������&2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d:::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
/__inference_sequential_1_layer_call_fn_14631012
embedding_1_input
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������&*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_146310032
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������&2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:���������
+
_user_specified_nameembedding_1_input
�
�
/__inference_sequential_1_layer_call_fn_14631109

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������&*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_146310272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������&2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_lambda_1_layer_call_and_return_conditional_losses_14630933

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:���������d2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
J__inference_sequential_1_layer_call_and_return_conditional_losses_14631087

inputs)
%embedding_1_embedding_lookup_14631072*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity�u
embedding_1/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding_1/Cast�
embedding_1/embedding_lookupResourceGather%embedding_1_embedding_lookup_14631072embedding_1/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_1/embedding_lookup/14631072*+
_output_shapes
:���������d*
dtype02
embedding_1/embedding_lookup�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_1/embedding_lookup/14631072*+
_output_shapes
:���������d2'
%embedding_1/embedding_lookup/Identity�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d2)
'embedding_1/embedding_lookup/Identity_1�
lambda_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2!
lambda_1/Mean/reduction_indices�
lambda_1/MeanMean0embedding_1/embedding_lookup/Identity_1:output:0(lambda_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������d2
lambda_1/Mean�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d�&*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMullambda_1/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������&2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�&*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������&2
dense_1/BiasAddz
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������&2
dense_1/Softmaxn
IdentityIdentitydense_1/Softmax:softmax:0*
T0*(
_output_shapes
:����������&2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������::::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
!__inference__traced_save_14631200
file_prefix5
1savev2_embedding_1_embeddings_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_25673868874e49d2afe3821dfcb3ead7/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_1_embeddings_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*4
_input_shapes#
!: :	�&d:	d�&:�&: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�&d:%!

_output_shapes
:	d�&:!

_output_shapes	
:�&:

_output_shapes
: 
�

*__inference_dense_1_layer_call_fn_14631168

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_146309572
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������&2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
/__inference_sequential_1_layer_call_fn_14631036
embedding_1_input
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallembedding_1_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������&*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_146310272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������&2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:���������
+
_user_specified_nameembedding_1_input
�
G
+__inference_lambda_1_layer_call_fn_14631148

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_146309332
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
/__inference_sequential_1_layer_call_fn_14631098

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������&*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_1_layer_call_and_return_conditional_losses_146310032
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������&2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_sequential_1_layer_call_and_return_conditional_losses_14631003

inputs
embedding_1_14630993
dense_1_14630997
dense_1_14630999
identity��dense_1/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_14630993*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_embedding_1_layer_call_and_return_conditional_losses_146309092%
#embedding_1/StatefulPartitionedCall�
lambda_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lambda_1_layer_call_and_return_conditional_losses_146309272
lambda_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall!lambda_1/PartitionedCall:output:0dense_1_14630997dense_1_14630999*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_146309572!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*
T0*(
_output_shapes
:����������&2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������:::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_embedding_1_layer_call_and_return_conditional_losses_14631119

inputs
embedding_lookup_14631113
identity�]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_14631113Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/14631113*+
_output_shapes
:���������d*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/14631113*+
_output_shapes
:���������d2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d2
embedding_lookup/Identity_1|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������d2

Identity"
identityIdentity:output:0**
_input_shapes
:���������::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
O
embedding_1_input:
#serving_default_embedding_1_input:0���������<
dense_11
StatefulPartitionedCall:0����������&tensorflow/serving/predict:�x
�!
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
-_default_save_signature
.__call__
*/&call_and_return_all_conditional_losses"�
_tf_keras_sequential�{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_1_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "input_dim": 4930, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 4}}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAQwAAAHMOAAAAdABqAXwAZAFkAo0CUwApA07pAQAAACkB2gRh\neGlzKQLaAUvaBG1lYW4pAdoBeKkAcgYAAAD6HzxpcHl0aG9uLWlucHV0LTM1LTIxMGQwMGI0NmQ4\nYz7aCDxsYW1iZGE+BwAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": [100]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 4930, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_1_input"}}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "input_dim": 4930, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 4}}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAQwAAAHMOAAAAdABqAXwAZAFkAo0CUwApA07pAQAAACkB2gRh\neGlzKQLaAUvaBG1lYW4pAdoBeKkAcgYAAAD6HzxpcHl0aG9uLWlucHV0LTM1LTIxMGQwMGI0NmQ4\nYz7aCDxsYW1iZGE+BwAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": [100]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 4930, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.001, "decay": 0.0, "rho": 0.9, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
�


embeddings
trainable_variables
regularization_losses
	variables
	keras_api
0__call__
*1&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "input_dim": 4930, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 4}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
�
trainable_variables
regularization_losses
	variables
	keras_api
2__call__
*3&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Lambda", "name": "lambda_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAQwAAAHMOAAAAdABqAXwAZAFkAo0CUwApA07pAQAAACkB2gRh\neGlzKQLaAUvaBG1lYW4pAdoBeKkAcgYAAAD6HzxpcHl0aG9uLWlucHV0LTM1LTIxMGQwMGI0NmQ4\nYz7aCDxsYW1iZGE+BwAAAPMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": {"class_name": "__tuple__", "items": [100]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
4__call__
*5&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 4930, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
"
	optimizer
5

0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5

0
1
2"
trackable_list_wrapper
�
metrics

layers
trainable_variables
layer_metrics
non_trainable_variables
layer_regularization_losses
regularization_losses
	variables
.__call__
-_default_save_signature
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
,
6serving_default"
signature_map
):'	�&d2embedding_1/embeddings
'

0"
trackable_list_wrapper
 "
trackable_list_wrapper
'

0"
trackable_list_wrapper
�
metrics

layers
trainable_variables
 layer_metrics
!non_trainable_variables
"layer_regularization_losses
regularization_losses
	variables
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
#metrics

$layers
trainable_variables
%layer_metrics
&non_trainable_variables
'layer_regularization_losses
regularization_losses
	variables
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
!:	d�&2dense_1/kernel
:�&2dense_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
(metrics

)layers
trainable_variables
*layer_metrics
+non_trainable_variables
,layer_regularization_losses
regularization_losses
	variables
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
#__inference__wrapped_model_14630895�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *0�-
+�(
embedding_1_input���������
�2�
/__inference_sequential_1_layer_call_fn_14631109
/__inference_sequential_1_layer_call_fn_14631012
/__inference_sequential_1_layer_call_fn_14631098
/__inference_sequential_1_layer_call_fn_14631036�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_sequential_1_layer_call_and_return_conditional_losses_14630987
J__inference_sequential_1_layer_call_and_return_conditional_losses_14631087
J__inference_sequential_1_layer_call_and_return_conditional_losses_14630974
J__inference_sequential_1_layer_call_and_return_conditional_losses_14631068�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
.__inference_embedding_1_layer_call_fn_14631126�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_embedding_1_layer_call_and_return_conditional_losses_14631119�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_lambda_1_layer_call_fn_14631148
+__inference_lambda_1_layer_call_fn_14631143�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_lambda_1_layer_call_and_return_conditional_losses_14631132
F__inference_lambda_1_layer_call_and_return_conditional_losses_14631138�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_dense_1_layer_call_fn_14631168�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_1_layer_call_and_return_conditional_losses_14631159�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
?B=
&__inference_signature_wrapper_14631049embedding_1_input�
#__inference__wrapped_model_14630895u
:�7
0�-
+�(
embedding_1_input���������
� "2�/
-
dense_1"�
dense_1����������&�
E__inference_dense_1_layer_call_and_return_conditional_losses_14631159]/�,
%�"
 �
inputs���������d
� "&�#
�
0����������&
� ~
*__inference_dense_1_layer_call_fn_14631168P/�,
%�"
 �
inputs���������d
� "�����������&�
I__inference_embedding_1_layer_call_and_return_conditional_losses_14631119_
/�,
%�"
 �
inputs���������
� ")�&
�
0���������d
� �
.__inference_embedding_1_layer_call_fn_14631126R
/�,
%�"
 �
inputs���������
� "����������d�
F__inference_lambda_1_layer_call_and_return_conditional_losses_14631132d;�8
1�.
$�!
inputs���������d

 
p
� "%�"
�
0���������d
� �
F__inference_lambda_1_layer_call_and_return_conditional_losses_14631138d;�8
1�.
$�!
inputs���������d

 
p 
� "%�"
�
0���������d
� �
+__inference_lambda_1_layer_call_fn_14631143W;�8
1�.
$�!
inputs���������d

 
p
� "����������d�
+__inference_lambda_1_layer_call_fn_14631148W;�8
1�.
$�!
inputs���������d

 
p 
� "����������d�
J__inference_sequential_1_layer_call_and_return_conditional_losses_14630974q
B�?
8�5
+�(
embedding_1_input���������
p

 
� "&�#
�
0����������&
� �
J__inference_sequential_1_layer_call_and_return_conditional_losses_14630987q
B�?
8�5
+�(
embedding_1_input���������
p 

 
� "&�#
�
0����������&
� �
J__inference_sequential_1_layer_call_and_return_conditional_losses_14631068f
7�4
-�*
 �
inputs���������
p

 
� "&�#
�
0����������&
� �
J__inference_sequential_1_layer_call_and_return_conditional_losses_14631087f
7�4
-�*
 �
inputs���������
p 

 
� "&�#
�
0����������&
� �
/__inference_sequential_1_layer_call_fn_14631012d
B�?
8�5
+�(
embedding_1_input���������
p

 
� "�����������&�
/__inference_sequential_1_layer_call_fn_14631036d
B�?
8�5
+�(
embedding_1_input���������
p 

 
� "�����������&�
/__inference_sequential_1_layer_call_fn_14631098Y
7�4
-�*
 �
inputs���������
p

 
� "�����������&�
/__inference_sequential_1_layer_call_fn_14631109Y
7�4
-�*
 �
inputs���������
p 

 
� "�����������&�
&__inference_signature_wrapper_14631049�
O�L
� 
E�B
@
embedding_1_input+�(
embedding_1_input���������"2�/
-
dense_1"�
dense_1����������&