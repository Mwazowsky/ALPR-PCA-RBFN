Цв
°т
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
7
Square
x"T
y"T"
Ttype:
2	
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ки
w
hidden/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Fи*
shared_namehidden/kernel
p
!hidden/kernel/Read/ReadVariableOpReadVariableOphidden/kernel*
_output_shapes
:	Fи*
dtype0
o
hidden/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:и*
shared_namehidden/bias
h
hidden/bias/Read/ReadVariableOpReadVariableOphidden/bias*
_output_shapes	
:и*
dtype0
С
batch_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:и*-
shared_namebatch_normalization_18/gamma
К
0batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_18/gamma*
_output_shapes	
:и*
dtype0
П
batch_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:и*,
shared_namebatch_normalization_18/beta
И
/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_18/beta*
_output_shapes	
:и*
dtype0
Э
"batch_normalization_18/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:и*3
shared_name$"batch_normalization_18/moving_mean
Ц
6batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
_output_shapes	
:и*
dtype0
•
&batch_normalization_18/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:и*7
shared_name(&batch_normalization_18/moving_variance
Ю
:batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
_output_shapes	
:и*
dtype0
q

out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и$*
shared_name
out/kernel
j
out/kernel/Read/ReadVariableOpReadVariableOp
out/kernel*
_output_shapes
:	и$*
dtype0
h
out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_name
out/bias
a
out/bias/Read/ReadVariableOpReadVariableOpout/bias*
_output_shapes
:$*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
Е
Adam/hidden/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Fи*%
shared_nameAdam/hidden/kernel/m
~
(Adam/hidden/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden/kernel/m*
_output_shapes
:	Fи*
dtype0
}
Adam/hidden/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:и*#
shared_nameAdam/hidden/bias/m
v
&Adam/hidden/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden/bias/m*
_output_shapes	
:и*
dtype0
Я
#Adam/batch_normalization_18/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:и*4
shared_name%#Adam/batch_normalization_18/gamma/m
Ш
7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/m*
_output_shapes	
:и*
dtype0
Э
"Adam/batch_normalization_18/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:и*3
shared_name$"Adam/batch_normalization_18/beta/m
Ц
6Adam/batch_normalization_18/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/m*
_output_shapes	
:и*
dtype0

Adam/out/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и$*"
shared_nameAdam/out/kernel/m
x
%Adam/out/kernel/m/Read/ReadVariableOpReadVariableOpAdam/out/kernel/m*
_output_shapes
:	и$*
dtype0
v
Adam/out/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$* 
shared_nameAdam/out/bias/m
o
#Adam/out/bias/m/Read/ReadVariableOpReadVariableOpAdam/out/bias/m*
_output_shapes
:$*
dtype0
Е
Adam/hidden/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Fи*%
shared_nameAdam/hidden/kernel/v
~
(Adam/hidden/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden/kernel/v*
_output_shapes
:	Fи*
dtype0
}
Adam/hidden/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:и*#
shared_nameAdam/hidden/bias/v
v
&Adam/hidden/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden/bias/v*
_output_shapes	
:и*
dtype0
Я
#Adam/batch_normalization_18/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:и*4
shared_name%#Adam/batch_normalization_18/gamma/v
Ш
7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/v*
_output_shapes	
:и*
dtype0
Э
"Adam/batch_normalization_18/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:и*3
shared_name$"Adam/batch_normalization_18/beta/v
Ц
6Adam/batch_normalization_18/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/v*
_output_shapes	
:и*
dtype0

Adam/out/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и$*"
shared_nameAdam/out/kernel/v
x
%Adam/out/kernel/v/Read/ReadVariableOpReadVariableOpAdam/out/kernel/v*
_output_shapes
:	и$*
dtype0
v
Adam/out/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$* 
shared_nameAdam/out/bias/v
o
#Adam/out/bias/v/Read/ReadVariableOpReadVariableOpAdam/out/bias/v*
_output_shapes
:$*
dtype0

NoOpNoOp
Щ4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*‘3
value 3B«3 Bј3
ќ
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¶

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
•
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses* 
’
axis
	gamma
beta
 moving_mean
!moving_variance
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
¶

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses*
∞
0iter

1beta_1

2beta_2
	3decay
4learning_ratemZm[m\m](m^)m_v`vavbvc(vd)ve*
<
0
1
2
3
 4
!5
(6
)7*
.
0
1
2
3
(4
)5*
* 
∞
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

:serving_default* 
]W
VARIABLE_VALUEhidden/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEhidden/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
У
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
С
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_18/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_18/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_18/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_18/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
0
1
 2
!3*

0
1*
* 
У
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 
ZT
VARIABLE_VALUE
out/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEout/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

(0
)1*

(0
)1*
* 
У
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*

 0
!1*
 
0
1
2
3*

O0
P1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

 0
!1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	Qtotal
	Rcount
S	variables
T	keras_api*
H
	Utotal
	Vcount
W
_fn_kwargs
X	variables
Y	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Q0
R1*

S	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

U0
V1*

X	variables*
Аz
VARIABLE_VALUEAdam/hidden/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/hidden/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_18/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_18/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/out/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/out/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/hidden/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/hidden/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_18/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_18/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/out/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/out/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_hidden_inputPlaceholder*'
_output_shapes
:€€€€€€€€€F*
dtype0*
shape:€€€€€€€€€F
Б
StatefulPartitionedCallStatefulPartitionedCallserving_default_hidden_inputhidden/kernelhidden/bias&batch_normalization_18/moving_variancebatch_normalization_18/gamma"batch_normalization_18/moving_meanbatch_normalization_18/beta
out/kernelout/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_2171127
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
…
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!hidden/kernel/Read/ReadVariableOphidden/bias/Read/ReadVariableOp0batch_normalization_18/gamma/Read/ReadVariableOp/batch_normalization_18/beta/Read/ReadVariableOp6batch_normalization_18/moving_mean/Read/ReadVariableOp:batch_normalization_18/moving_variance/Read/ReadVariableOpout/kernel/Read/ReadVariableOpout/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/hidden/kernel/m/Read/ReadVariableOp&Adam/hidden/bias/m/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_18/beta/m/Read/ReadVariableOp%Adam/out/kernel/m/Read/ReadVariableOp#Adam/out/bias/m/Read/ReadVariableOp(Adam/hidden/kernel/v/Read/ReadVariableOp&Adam/hidden/bias/v/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_18/beta/v/Read/ReadVariableOp%Adam/out/kernel/v/Read/ReadVariableOp#Adam/out/bias/v/Read/ReadVariableOpConst**
Tin#
!2	*
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
GPU 2J 8В *)
f$R"
 __inference__traced_save_2171387
А
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden/kernelhidden/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_variance
out/kernelout/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/hidden/kernel/mAdam/hidden/bias/m#Adam/batch_normalization_18/gamma/m"Adam/batch_normalization_18/beta/mAdam/out/kernel/mAdam/out/bias/mAdam/hidden/kernel/vAdam/hidden/bias/v#Adam/batch_normalization_18/gamma/v"Adam/batch_normalization_18/beta/vAdam/out/kernel/vAdam/out/bias/v*)
Tin"
 2*
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
GPU 2J 8В *,
f'R%
#__inference__traced_restore_2171484Уд
К
ц
C__inference_hidden_layer_call_and_return_conditional_losses_2170722

inputs1
matmul_readvariableop_resource:	Fи.
biasadd_readvariableop_resource:	и
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Fи*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€иs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:и*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€иU
SquareSquareBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€иJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  АњY
mulMulmul/x:output:0
Square:y:0*
T0*(
_output_shapes
:€€€€€€€€€иF
ExpExpmul:z:0*
T0*(
_output_shapes
:€€€€€€€€€иW
IdentityIdentityExp:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€иw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€F: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€F
 
_user_specified_nameinputs
І
H
,__inference_dropout_18_layer_call_fn_2171155

inputs
identity≥
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_2170733a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€и"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€и:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
р9
ц
"__inference__wrapped_model_2170619
hidden_inputF
3sequential_18_hidden_matmul_readvariableop_resource:	FиC
4sequential_18_hidden_biasadd_readvariableop_resource:	иU
Fsequential_18_batch_normalization_18_batchnorm_readvariableop_resource:	иY
Jsequential_18_batch_normalization_18_batchnorm_mul_readvariableop_resource:	иW
Hsequential_18_batch_normalization_18_batchnorm_readvariableop_1_resource:	иW
Hsequential_18_batch_normalization_18_batchnorm_readvariableop_2_resource:	иC
0sequential_18_out_matmul_readvariableop_resource:	и$?
1sequential_18_out_biasadd_readvariableop_resource:$
identityИҐ=sequential_18/batch_normalization_18/batchnorm/ReadVariableOpҐ?sequential_18/batch_normalization_18/batchnorm/ReadVariableOp_1Ґ?sequential_18/batch_normalization_18/batchnorm/ReadVariableOp_2ҐAsequential_18/batch_normalization_18/batchnorm/mul/ReadVariableOpҐ+sequential_18/hidden/BiasAdd/ReadVariableOpҐ*sequential_18/hidden/MatMul/ReadVariableOpҐ(sequential_18/out/BiasAdd/ReadVariableOpҐ'sequential_18/out/MatMul/ReadVariableOpЯ
*sequential_18/hidden/MatMul/ReadVariableOpReadVariableOp3sequential_18_hidden_matmul_readvariableop_resource*
_output_shapes
:	Fи*
dtype0Ъ
sequential_18/hidden/MatMulMatMulhidden_input2sequential_18/hidden/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€иЭ
+sequential_18/hidden/BiasAdd/ReadVariableOpReadVariableOp4sequential_18_hidden_biasadd_readvariableop_resource*
_output_shapes	
:и*
dtype0ґ
sequential_18/hidden/BiasAddBiasAdd%sequential_18/hidden/MatMul:product:03sequential_18/hidden/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€и
sequential_18/hidden/SquareSquare%sequential_18/hidden/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€и_
sequential_18/hidden/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  АњШ
sequential_18/hidden/mulMul#sequential_18/hidden/mul/x:output:0sequential_18/hidden/Square:y:0*
T0*(
_output_shapes
:€€€€€€€€€иp
sequential_18/hidden/ExpExpsequential_18/hidden/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€и~
!sequential_18/dropout_18/IdentityIdentitysequential_18/hidden/Exp:y:0*
T0*(
_output_shapes
:€€€€€€€€€иЅ
=sequential_18/batch_normalization_18/batchnorm/ReadVariableOpReadVariableOpFsequential_18_batch_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:и*
dtype0y
4sequential_18/batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:з
2sequential_18/batch_normalization_18/batchnorm/addAddV2Esequential_18/batch_normalization_18/batchnorm/ReadVariableOp:value:0=sequential_18/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:иЫ
4sequential_18/batch_normalization_18/batchnorm/RsqrtRsqrt6sequential_18/batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:и…
Asequential_18/batch_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOpJsequential_18_batch_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:и*
dtype0д
2sequential_18/batch_normalization_18/batchnorm/mulMul8sequential_18/batch_normalization_18/batchnorm/Rsqrt:y:0Isequential_18/batch_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:и“
4sequential_18/batch_normalization_18/batchnorm/mul_1Mul*sequential_18/dropout_18/Identity:output:06sequential_18/batch_normalization_18/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€и≈
?sequential_18/batch_normalization_18/batchnorm/ReadVariableOp_1ReadVariableOpHsequential_18_batch_normalization_18_batchnorm_readvariableop_1_resource*
_output_shapes	
:и*
dtype0в
4sequential_18/batch_normalization_18/batchnorm/mul_2MulGsequential_18/batch_normalization_18/batchnorm/ReadVariableOp_1:value:06sequential_18/batch_normalization_18/batchnorm/mul:z:0*
T0*
_output_shapes	
:и≈
?sequential_18/batch_normalization_18/batchnorm/ReadVariableOp_2ReadVariableOpHsequential_18_batch_normalization_18_batchnorm_readvariableop_2_resource*
_output_shapes	
:и*
dtype0в
2sequential_18/batch_normalization_18/batchnorm/subSubGsequential_18/batch_normalization_18/batchnorm/ReadVariableOp_2:value:08sequential_18/batch_normalization_18/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ив
4sequential_18/batch_normalization_18/batchnorm/add_1AddV28sequential_18/batch_normalization_18/batchnorm/mul_1:z:06sequential_18/batch_normalization_18/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€иЩ
'sequential_18/out/MatMul/ReadVariableOpReadVariableOp0sequential_18_out_matmul_readvariableop_resource*
_output_shapes
:	и$*
dtype0њ
sequential_18/out/MatMulMatMul8sequential_18/batch_normalization_18/batchnorm/add_1:z:0/sequential_18/out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$Ц
(sequential_18/out/BiasAdd/ReadVariableOpReadVariableOp1sequential_18_out_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0ђ
sequential_18/out/BiasAddBiasAdd"sequential_18/out/MatMul:product:00sequential_18/out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$z
sequential_18/out/SoftmaxSoftmax"sequential_18/out/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$r
IdentityIdentity#sequential_18/out/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$ю
NoOpNoOp>^sequential_18/batch_normalization_18/batchnorm/ReadVariableOp@^sequential_18/batch_normalization_18/batchnorm/ReadVariableOp_1@^sequential_18/batch_normalization_18/batchnorm/ReadVariableOp_2B^sequential_18/batch_normalization_18/batchnorm/mul/ReadVariableOp,^sequential_18/hidden/BiasAdd/ReadVariableOp+^sequential_18/hidden/MatMul/ReadVariableOp)^sequential_18/out/BiasAdd/ReadVariableOp(^sequential_18/out/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€F: : : : : : : : 2~
=sequential_18/batch_normalization_18/batchnorm/ReadVariableOp=sequential_18/batch_normalization_18/batchnorm/ReadVariableOp2В
?sequential_18/batch_normalization_18/batchnorm/ReadVariableOp_1?sequential_18/batch_normalization_18/batchnorm/ReadVariableOp_12В
?sequential_18/batch_normalization_18/batchnorm/ReadVariableOp_2?sequential_18/batch_normalization_18/batchnorm/ReadVariableOp_22Ж
Asequential_18/batch_normalization_18/batchnorm/mul/ReadVariableOpAsequential_18/batch_normalization_18/batchnorm/mul/ReadVariableOp2Z
+sequential_18/hidden/BiasAdd/ReadVariableOp+sequential_18/hidden/BiasAdd/ReadVariableOp2X
*sequential_18/hidden/MatMul/ReadVariableOp*sequential_18/hidden/MatMul/ReadVariableOp2T
(sequential_18/out/BiasAdd/ReadVariableOp(sequential_18/out/BiasAdd/ReadVariableOp2R
'sequential_18/out/MatMul/ReadVariableOp'sequential_18/out/MatMul/ReadVariableOp:U Q
'
_output_shapes
:€€€€€€€€€F
&
_user_specified_namehidden_input
«	
љ
/__inference_sequential_18_layer_call_fn_2171007

inputs
unknown:	Fи
	unknown_0:	и
	unknown_1:	и
	unknown_2:	и
	unknown_3:	и
	unknown_4:	и
	unknown_5:	и$
	unknown_6:$
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_18_layer_call_and_return_conditional_losses_2170871o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€F: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€F
 
_user_specified_nameinputs
±%
р
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2170690

inputs6
'assignmovingavg_readvariableop_resource:	и8
)assignmovingavg_1_readvariableop_resource:	и4
%batchnorm_mul_readvariableop_resource:	и0
!batchnorm_readvariableop_resource:	и
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	и*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	иИ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€иl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	и*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:и*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:и*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:и*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:иy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:иђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:и*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:и
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:иі
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:иQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:и
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:и*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:иd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€иi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:иw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:и*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:иs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€иc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ик
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€и: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
ў	
√
/__inference_sequential_18_layer_call_fn_2170911
hidden_input
unknown:	Fи
	unknown_0:	и
	unknown_1:	и
	unknown_2:	и
	unknown_3:	и
	unknown_4:	и
	unknown_5:	и$
	unknown_6:$
identityИҐStatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallhidden_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_18_layer_call_and_return_conditional_losses_2170871o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€F: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:€€€€€€€€€F
&
_user_specified_namehidden_input
•
љ
J__inference_sequential_18_layer_call_and_return_conditional_losses_2170959
hidden_input!
hidden_2170938:	Fи
hidden_2170940:	и-
batch_normalization_18_2170944:	и-
batch_normalization_18_2170946:	и-
batch_normalization_18_2170948:	и-
batch_normalization_18_2170950:	и
out_2170953:	и$
out_2170955:$
identityИҐ.batch_normalization_18/StatefulPartitionedCallҐ"dropout_18/StatefulPartitionedCallҐhidden/StatefulPartitionedCallҐout/StatefulPartitionedCallт
hidden/StatefulPartitionedCallStatefulPartitionedCallhidden_inputhidden_2170938hidden_2170940*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_layer_call_and_return_conditional_losses_2170722п
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_2170811У
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0batch_normalization_18_2170944batch_normalization_18_2170946batch_normalization_18_2170948batch_normalization_18_2170950*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2170690Р
out/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0out_2170953out_2170955*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_out_layer_call_and_return_conditional_losses_2170755s
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$џ
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€F: : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:U Q
'
_output_shapes
:€€€€€€€€€F
&
_user_specified_namehidden_input
ш.
Є
J__inference_sequential_18_layer_call_and_return_conditional_losses_2171045

inputs8
%hidden_matmul_readvariableop_resource:	Fи5
&hidden_biasadd_readvariableop_resource:	иG
8batch_normalization_18_batchnorm_readvariableop_resource:	иK
<batch_normalization_18_batchnorm_mul_readvariableop_resource:	иI
:batch_normalization_18_batchnorm_readvariableop_1_resource:	иI
:batch_normalization_18_batchnorm_readvariableop_2_resource:	и5
"out_matmul_readvariableop_resource:	и$1
#out_biasadd_readvariableop_resource:$
identityИҐ/batch_normalization_18/batchnorm/ReadVariableOpҐ1batch_normalization_18/batchnorm/ReadVariableOp_1Ґ1batch_normalization_18/batchnorm/ReadVariableOp_2Ґ3batch_normalization_18/batchnorm/mul/ReadVariableOpҐhidden/BiasAdd/ReadVariableOpҐhidden/MatMul/ReadVariableOpҐout/BiasAdd/ReadVariableOpҐout/MatMul/ReadVariableOpГ
hidden/MatMul/ReadVariableOpReadVariableOp%hidden_matmul_readvariableop_resource*
_output_shapes
:	Fи*
dtype0x
hidden/MatMulMatMulinputs$hidden/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€иБ
hidden/BiasAdd/ReadVariableOpReadVariableOp&hidden_biasadd_readvariableop_resource*
_output_shapes	
:и*
dtype0М
hidden/BiasAddBiasAddhidden/MatMul:product:0%hidden/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€иc
hidden/SquareSquarehidden/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€иQ
hidden/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  Ањn

hidden/mulMulhidden/mul/x:output:0hidden/Square:y:0*
T0*(
_output_shapes
:€€€€€€€€€иT

hidden/ExpExphidden/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€иb
dropout_18/IdentityIdentityhidden/Exp:y:0*
T0*(
_output_shapes
:€€€€€€€€€и•
/batch_normalization_18/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:и*
dtype0k
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:љ
$batch_normalization_18/batchnorm/addAddV27batch_normalization_18/batchnorm/ReadVariableOp:value:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:и
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:и≠
3batch_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:и*
dtype0Ї
$batch_normalization_18/batchnorm/mulMul*batch_normalization_18/batchnorm/Rsqrt:y:0;batch_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:и®
&batch_normalization_18/batchnorm/mul_1Muldropout_18/Identity:output:0(batch_normalization_18/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€и©
1batch_normalization_18/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_18_batchnorm_readvariableop_1_resource*
_output_shapes	
:и*
dtype0Є
&batch_normalization_18/batchnorm/mul_2Mul9batch_normalization_18/batchnorm/ReadVariableOp_1:value:0(batch_normalization_18/batchnorm/mul:z:0*
T0*
_output_shapes	
:и©
1batch_normalization_18/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_18_batchnorm_readvariableop_2_resource*
_output_shapes	
:и*
dtype0Є
$batch_normalization_18/batchnorm/subSub9batch_normalization_18/batchnorm/ReadVariableOp_2:value:0*batch_normalization_18/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:иЄ
&batch_normalization_18/batchnorm/add_1AddV2*batch_normalization_18/batchnorm/mul_1:z:0(batch_normalization_18/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€и}
out/MatMul/ReadVariableOpReadVariableOp"out_matmul_readvariableop_resource*
_output_shapes
:	и$*
dtype0Х

out/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0!out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$z
out/BiasAdd/ReadVariableOpReadVariableOp#out_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0В
out/BiasAddBiasAddout/MatMul:product:0"out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$^
out/SoftmaxSoftmaxout/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$d
IdentityIdentityout/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$О
NoOpNoOp0^batch_normalization_18/batchnorm/ReadVariableOp2^batch_normalization_18/batchnorm/ReadVariableOp_12^batch_normalization_18/batchnorm/ReadVariableOp_24^batch_normalization_18/batchnorm/mul/ReadVariableOp^hidden/BiasAdd/ReadVariableOp^hidden/MatMul/ReadVariableOp^out/BiasAdd/ReadVariableOp^out/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€F: : : : : : : : 2b
/batch_normalization_18/batchnorm/ReadVariableOp/batch_normalization_18/batchnorm/ReadVariableOp2f
1batch_normalization_18/batchnorm/ReadVariableOp_11batch_normalization_18/batchnorm/ReadVariableOp_12f
1batch_normalization_18/batchnorm/ReadVariableOp_21batch_normalization_18/batchnorm/ReadVariableOp_22j
3batch_normalization_18/batchnorm/mul/ReadVariableOp3batch_normalization_18/batchnorm/mul/ReadVariableOp2>
hidden/BiasAdd/ReadVariableOphidden/BiasAdd/ReadVariableOp2<
hidden/MatMul/ReadVariableOphidden/MatMul/ReadVariableOp28
out/BiasAdd/ReadVariableOpout/BiasAdd/ReadVariableOp26
out/MatMul/ReadVariableOpout/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€F
 
_user_specified_nameinputs
Р@
Љ
 __inference__traced_save_2171387
file_prefix,
(savev2_hidden_kernel_read_readvariableop*
&savev2_hidden_bias_read_readvariableop;
7savev2_batch_normalization_18_gamma_read_readvariableop:
6savev2_batch_normalization_18_beta_read_readvariableopA
=savev2_batch_normalization_18_moving_mean_read_readvariableopE
Asavev2_batch_normalization_18_moving_variance_read_readvariableop)
%savev2_out_kernel_read_readvariableop'
#savev2_out_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_hidden_kernel_m_read_readvariableop1
-savev2_adam_hidden_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_m_read_readvariableop0
,savev2_adam_out_kernel_m_read_readvariableop.
*savev2_adam_out_bias_m_read_readvariableop3
/savev2_adam_hidden_kernel_v_read_readvariableop1
-savev2_adam_hidden_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_v_read_readvariableop0
,savev2_adam_out_kernel_v_read_readvariableop.
*savev2_adam_out_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: р
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Щ
valueПBМB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH©
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ®
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_hidden_kernel_read_readvariableop&savev2_hidden_bias_read_readvariableop7savev2_batch_normalization_18_gamma_read_readvariableop6savev2_batch_normalization_18_beta_read_readvariableop=savev2_batch_normalization_18_moving_mean_read_readvariableopAsavev2_batch_normalization_18_moving_variance_read_readvariableop%savev2_out_kernel_read_readvariableop#savev2_out_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_hidden_kernel_m_read_readvariableop-savev2_adam_hidden_bias_m_read_readvariableop>savev2_adam_batch_normalization_18_gamma_m_read_readvariableop=savev2_adam_batch_normalization_18_beta_m_read_readvariableop,savev2_adam_out_kernel_m_read_readvariableop*savev2_adam_out_bias_m_read_readvariableop/savev2_adam_hidden_kernel_v_read_readvariableop-savev2_adam_hidden_bias_v_read_readvariableop>savev2_adam_batch_normalization_18_gamma_v_read_readvariableop=savev2_adam_batch_normalization_18_beta_v_read_readvariableop,savev2_adam_out_kernel_v_read_readvariableop*savev2_adam_out_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ћ
_input_shapesЇ
Ј: :	Fи:и:и:и:и:и:	и$:$: : : : : : : : : :	Fи:и:и:и:	и$:$:	Fи:и:и:и:	и$:$: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	Fи:!

_output_shapes	
:и:!

_output_shapes	
:и:!

_output_shapes	
:и:!

_output_shapes	
:и:!

_output_shapes	
:и:%!

_output_shapes
:	и$: 

_output_shapes
:$:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	Fи:!

_output_shapes	
:и:!

_output_shapes	
:и:!

_output_shapes	
:и:%!

_output_shapes
:	и$: 

_output_shapes
:$:%!

_output_shapes
:	Fи:!

_output_shapes	
:и:!

_output_shapes	
:и:!

_output_shapes	
:и:%!

_output_shapes
:	и$: 

_output_shapes
:$:

_output_shapes
: 
э	
f
G__inference_dropout_18_layer_call_and_return_conditional_losses_2171177

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€иC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€и*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€иp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€иj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€иZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€и"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€и:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
і
„
8__inference_batch_normalization_18_layer_call_fn_2171190

inputs
unknown:	и
	unknown_0:	и
	unknown_1:	и
	unknown_2:	и
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2170643p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€и`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€и: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
±%
р
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2171257

inputs6
'assignmovingavg_readvariableop_resource:	и8
)assignmovingavg_1_readvariableop_resource:	и4
%batchnorm_mul_readvariableop_resource:	и0
!batchnorm_readvariableop_resource:	и
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	и*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	иИ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€иl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	и*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:и*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:и*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:и*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:иy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:иђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:и*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:и
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:иі
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:иQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:и
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:и*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:иd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€иi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:иw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:и*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:иs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€иc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ик
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€и: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
У
Ј
J__inference_sequential_18_layer_call_and_return_conditional_losses_2170871

inputs!
hidden_2170850:	Fи
hidden_2170852:	и-
batch_normalization_18_2170856:	и-
batch_normalization_18_2170858:	и-
batch_normalization_18_2170860:	и-
batch_normalization_18_2170862:	и
out_2170865:	и$
out_2170867:$
identityИҐ.batch_normalization_18/StatefulPartitionedCallҐ"dropout_18/StatefulPartitionedCallҐhidden/StatefulPartitionedCallҐout/StatefulPartitionedCallм
hidden/StatefulPartitionedCallStatefulPartitionedCallinputshidden_2170850hidden_2170852*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_layer_call_and_return_conditional_losses_2170722п
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_2170811У
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0batch_normalization_18_2170856batch_normalization_18_2170858batch_normalization_18_2170860batch_normalization_18_2170862*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2170690Р
out/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0out_2170865out_2170867*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_out_layer_call_and_return_conditional_losses_2170755s
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$џ
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€F: : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€F
 
_user_specified_nameinputs
џ	
√
/__inference_sequential_18_layer_call_fn_2170781
hidden_input
unknown:	Fи
	unknown_0:	и
	unknown_1:	и
	unknown_2:	и
	unknown_3:	и
	unknown_4:	и
	unknown_5:	и$
	unknown_6:$
identityИҐStatefulPartitionedCall≥
StatefulPartitionedCallStatefulPartitionedCallhidden_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_18_layer_call_and_return_conditional_losses_2170762o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€F: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:€€€€€€€€€F
&
_user_specified_namehidden_input
†

т
@__inference_out_layer_call_and_return_conditional_losses_2171277

inputs1
matmul_readvariableop_resource:	и$-
biasadd_readvariableop_resource:$
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	и$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€и: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
й
Т
J__inference_sequential_18_layer_call_and_return_conditional_losses_2170762

inputs!
hidden_2170723:	Fи
hidden_2170725:	и-
batch_normalization_18_2170735:	и-
batch_normalization_18_2170737:	и-
batch_normalization_18_2170739:	и-
batch_normalization_18_2170741:	и
out_2170756:	и$
out_2170758:$
identityИҐ.batch_normalization_18/StatefulPartitionedCallҐhidden/StatefulPartitionedCallҐout/StatefulPartitionedCallм
hidden/StatefulPartitionedCallStatefulPartitionedCallinputshidden_2170723hidden_2170725*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_layer_call_and_return_conditional_losses_2170722я
dropout_18/PartitionedCallPartitionedCall'hidden/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_2170733Н
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0batch_normalization_18_2170735batch_normalization_18_2170737batch_normalization_18_2170739batch_normalization_18_2170741*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2170643Р
out/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0out_2170756out_2170758*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_out_layer_call_and_return_conditional_losses_2170755s
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$ґ
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€F: : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€F
 
_user_specified_nameinputs
ЋP
†
J__inference_sequential_18_layer_call_and_return_conditional_losses_2171104

inputs8
%hidden_matmul_readvariableop_resource:	Fи5
&hidden_biasadd_readvariableop_resource:	иM
>batch_normalization_18_assignmovingavg_readvariableop_resource:	иO
@batch_normalization_18_assignmovingavg_1_readvariableop_resource:	иK
<batch_normalization_18_batchnorm_mul_readvariableop_resource:	иG
8batch_normalization_18_batchnorm_readvariableop_resource:	и5
"out_matmul_readvariableop_resource:	и$1
#out_biasadd_readvariableop_resource:$
identityИҐ&batch_normalization_18/AssignMovingAvgҐ5batch_normalization_18/AssignMovingAvg/ReadVariableOpҐ(batch_normalization_18/AssignMovingAvg_1Ґ7batch_normalization_18/AssignMovingAvg_1/ReadVariableOpҐ/batch_normalization_18/batchnorm/ReadVariableOpҐ3batch_normalization_18/batchnorm/mul/ReadVariableOpҐhidden/BiasAdd/ReadVariableOpҐhidden/MatMul/ReadVariableOpҐout/BiasAdd/ReadVariableOpҐout/MatMul/ReadVariableOpГ
hidden/MatMul/ReadVariableOpReadVariableOp%hidden_matmul_readvariableop_resource*
_output_shapes
:	Fи*
dtype0x
hidden/MatMulMatMulinputs$hidden/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€иБ
hidden/BiasAdd/ReadVariableOpReadVariableOp&hidden_biasadd_readvariableop_resource*
_output_shapes	
:и*
dtype0М
hidden/BiasAddBiasAddhidden/MatMul:product:0%hidden/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€иc
hidden/SquareSquarehidden/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€иQ
hidden/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  Ањn

hidden/mulMulhidden/mul/x:output:0hidden/Square:y:0*
T0*(
_output_shapes
:€€€€€€€€€иT

hidden/ExpExphidden/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€и]
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Г
dropout_18/dropout/MulMulhidden/Exp:y:0!dropout_18/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€иV
dropout_18/dropout/ShapeShapehidden/Exp:y:0*
T0*
_output_shapes
:£
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€и*
dtype0f
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?»
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€иЖ
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€иЛ
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€и
5batch_normalization_18/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ƒ
#batch_normalization_18/moments/meanMeandropout_18/dropout/Mul_1:z:0>batch_normalization_18/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	и*
	keep_dims(У
+batch_normalization_18/moments/StopGradientStopGradient,batch_normalization_18/moments/mean:output:0*
T0*
_output_shapes
:	ић
0batch_normalization_18/moments/SquaredDifferenceSquaredDifferencedropout_18/dropout/Mul_1:z:04batch_normalization_18/moments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€иГ
9batch_normalization_18/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: д
'batch_normalization_18/moments/varianceMean4batch_normalization_18/moments/SquaredDifference:z:0Bbatch_normalization_18/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	и*
	keep_dims(Ь
&batch_normalization_18/moments/SqueezeSqueeze,batch_normalization_18/moments/mean:output:0*
T0*
_output_shapes	
:и*
squeeze_dims
 Ґ
(batch_normalization_18/moments/Squeeze_1Squeeze0batch_normalization_18/moments/variance:output:0*
T0*
_output_shapes	
:и*
squeeze_dims
 q
,batch_normalization_18/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<±
5batch_normalization_18/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_18_assignmovingavg_readvariableop_resource*
_output_shapes	
:и*
dtype0«
*batch_normalization_18/AssignMovingAvg/subSub=batch_normalization_18/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_18/moments/Squeeze:output:0*
T0*
_output_shapes	
:иЊ
*batch_normalization_18/AssignMovingAvg/mulMul.batch_normalization_18/AssignMovingAvg/sub:z:05batch_normalization_18/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:иИ
&batch_normalization_18/AssignMovingAvgAssignSubVariableOp>batch_normalization_18_assignmovingavg_readvariableop_resource.batch_normalization_18/AssignMovingAvg/mul:z:06^batch_normalization_18/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_18/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<µ
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_18_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:и*
dtype0Ќ
,batch_normalization_18/AssignMovingAvg_1/subSub?batch_normalization_18/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_18/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:иƒ
,batch_normalization_18/AssignMovingAvg_1/mulMul0batch_normalization_18/AssignMovingAvg_1/sub:z:07batch_normalization_18/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:иР
(batch_normalization_18/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_18_assignmovingavg_1_readvariableop_resource0batch_normalization_18/AssignMovingAvg_1/mul:z:08^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_18/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ј
$batch_normalization_18/batchnorm/addAddV21batch_normalization_18/moments/Squeeze_1:output:0/batch_normalization_18/batchnorm/add/y:output:0*
T0*
_output_shapes	
:и
&batch_normalization_18/batchnorm/RsqrtRsqrt(batch_normalization_18/batchnorm/add:z:0*
T0*
_output_shapes	
:и≠
3batch_normalization_18/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_18_batchnorm_mul_readvariableop_resource*
_output_shapes	
:и*
dtype0Ї
$batch_normalization_18/batchnorm/mulMul*batch_normalization_18/batchnorm/Rsqrt:y:0;batch_normalization_18/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:и®
&batch_normalization_18/batchnorm/mul_1Muldropout_18/dropout/Mul_1:z:0(batch_normalization_18/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€иЃ
&batch_normalization_18/batchnorm/mul_2Mul/batch_normalization_18/moments/Squeeze:output:0(batch_normalization_18/batchnorm/mul:z:0*
T0*
_output_shapes	
:и•
/batch_normalization_18/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_18_batchnorm_readvariableop_resource*
_output_shapes	
:и*
dtype0ґ
$batch_normalization_18/batchnorm/subSub7batch_normalization_18/batchnorm/ReadVariableOp:value:0*batch_normalization_18/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:иЄ
&batch_normalization_18/batchnorm/add_1AddV2*batch_normalization_18/batchnorm/mul_1:z:0(batch_normalization_18/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€и}
out/MatMul/ReadVariableOpReadVariableOp"out_matmul_readvariableop_resource*
_output_shapes
:	и$*
dtype0Х

out/MatMulMatMul*batch_normalization_18/batchnorm/add_1:z:0!out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$z
out/BiasAdd/ReadVariableOpReadVariableOp#out_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0В
out/BiasAddBiasAddout/MatMul:product:0"out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$^
out/SoftmaxSoftmaxout/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$d
IdentityIdentityout/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$м
NoOpNoOp'^batch_normalization_18/AssignMovingAvg6^batch_normalization_18/AssignMovingAvg/ReadVariableOp)^batch_normalization_18/AssignMovingAvg_18^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_18/batchnorm/ReadVariableOp4^batch_normalization_18/batchnorm/mul/ReadVariableOp^hidden/BiasAdd/ReadVariableOp^hidden/MatMul/ReadVariableOp^out/BiasAdd/ReadVariableOp^out/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€F: : : : : : : : 2P
&batch_normalization_18/AssignMovingAvg&batch_normalization_18/AssignMovingAvg2n
5batch_normalization_18/AssignMovingAvg/ReadVariableOp5batch_normalization_18/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_18/AssignMovingAvg_1(batch_normalization_18/AssignMovingAvg_12r
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_18/batchnorm/ReadVariableOp/batch_normalization_18/batchnorm/ReadVariableOp2j
3batch_normalization_18/batchnorm/mul/ReadVariableOp3batch_normalization_18/batchnorm/mul/ReadVariableOp2>
hidden/BiasAdd/ReadVariableOphidden/BiasAdd/ReadVariableOp2<
hidden/MatMul/ReadVariableOphidden/MatMul/ReadVariableOp28
out/BiasAdd/ReadVariableOpout/BiasAdd/ReadVariableOp26
out/MatMul/ReadVariableOpout/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€F
 
_user_specified_nameinputs
в
ґ
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2171223

inputs0
!batchnorm_readvariableop_resource:	и4
%batchnorm_mul_readvariableop_resource:	и2
#batchnorm_readvariableop_1_resource:	и2
#batchnorm_readvariableop_2_resource:	и
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:и*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:иQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:и
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:и*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:иd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€и{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:и*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:и{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:и*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:иs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€иc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€иЇ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€и: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
љ
У
%__inference_out_layer_call_fn_2171266

inputs
unknown:	и$
	unknown_0:$
identityИҐStatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_out_layer_call_and_return_conditional_losses_2170755o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€и: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
К
ц
C__inference_hidden_layer_call_and_return_conditional_losses_2171150

inputs1
matmul_readvariableop_resource:	Fи.
biasadd_readvariableop_resource:	и
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Fи*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€иs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:и*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€иU
SquareSquareBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€иJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  АњY
mulMulmul/x:output:0
Square:y:0*
T0*(
_output_shapes
:€€€€€€€€€иF
ExpExpmul:z:0*
T0*(
_output_shapes
:€€€€€€€€€иW
IdentityIdentityExp:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€иw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€F: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€F
 
_user_specified_nameinputs
щ
e
,__inference_dropout_18_layer_call_fn_2171160

inputs
identityИҐStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_2170811p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€и`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€и22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
э	
f
G__inference_dropout_18_layer_call_and_return_conditional_losses_2170811

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€иC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€и*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€иp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€иj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€иZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€и"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€и:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
в
ґ
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2170643

inputs0
!batchnorm_readvariableop_resource:	и4
%batchnorm_mul_readvariableop_resource:	и2
#batchnorm_readvariableop_1_resource:	и2
#batchnorm_readvariableop_2_resource:	и
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:и*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:иQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:и
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:и*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:иd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€и{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:и*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:и{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:и*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:иs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€иc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€иЇ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€и: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
≤
„
8__inference_batch_normalization_18_layer_call_fn_2171203

inputs
unknown:	и
	unknown_0:	и
	unknown_1:	и
	unknown_2:	и
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2170690p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€и`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€и: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
†

т
@__inference_out_layer_call_and_return_conditional_losses_2170755

inputs1
matmul_readvariableop_resource:	и$-
biasadd_readvariableop_resource:$
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	и$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€и: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
ё
e
G__inference_dropout_18_layer_call_and_return_conditional_losses_2171165

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€и\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€и"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€и:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
∞u
џ
#__inference__traced_restore_2171484
file_prefix1
assignvariableop_hidden_kernel:	Fи-
assignvariableop_1_hidden_bias:	и>
/assignvariableop_2_batch_normalization_18_gamma:	и=
.assignvariableop_3_batch_normalization_18_beta:	иD
5assignvariableop_4_batch_normalization_18_moving_mean:	иH
9assignvariableop_5_batch_normalization_18_moving_variance:	и0
assignvariableop_6_out_kernel:	и$)
assignvariableop_7_out_bias:$&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: ;
(assignvariableop_17_adam_hidden_kernel_m:	Fи5
&assignvariableop_18_adam_hidden_bias_m:	иF
7assignvariableop_19_adam_batch_normalization_18_gamma_m:	иE
6assignvariableop_20_adam_batch_normalization_18_beta_m:	и8
%assignvariableop_21_adam_out_kernel_m:	и$1
#assignvariableop_22_adam_out_bias_m:$;
(assignvariableop_23_adam_hidden_kernel_v:	Fи5
&assignvariableop_24_adam_hidden_bias_v:	иF
7assignvariableop_25_adam_batch_normalization_18_gamma_v:	иE
6assignvariableop_26_adam_batch_normalization_18_beta_v:	и8
%assignvariableop_27_adam_out_kernel_v:	и$1
#assignvariableop_28_adam_out_bias_v:$
identity_30ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9у
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Щ
valueПBМB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHђ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B µ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*М
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOpAssignVariableOpassignvariableop_hidden_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_1AssignVariableOpassignvariableop_1_hidden_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_18_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_18_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_18_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_18_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_6AssignVariableOpassignvariableop_6_out_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_7AssignVariableOpassignvariableop_7_out_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_hidden_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_hidden_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_19AssignVariableOp7assignvariableop_19_adam_batch_normalization_18_gamma_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_batch_normalization_18_beta_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_21AssignVariableOp%assignvariableop_21_adam_out_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_22AssignVariableOp#assignvariableop_22_adam_out_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_hidden_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_hidden_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_batch_normalization_18_gamma_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_batch_normalization_18_beta_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_out_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_28AssignVariableOp#assignvariableop_28_adam_out_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ќ
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_30IdentityIdentity_29:output:0^NoOp_1*
T0*
_output_shapes
: Ї
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_30Identity_30:output:0*O
_input_shapes>
<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ё
e
G__inference_dropout_18_layer_call_and_return_conditional_losses_2170733

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€и\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€и"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€и:P L
(
_output_shapes
:€€€€€€€€€и
 
_user_specified_nameinputs
…	
љ
/__inference_sequential_18_layer_call_fn_2170986

inputs
unknown:	Fи
	unknown_0:	и
	unknown_1:	и
	unknown_2:	и
	unknown_3:	и
	unknown_4:	и
	unknown_5:	и$
	unknown_6:$
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_18_layer_call_and_return_conditional_losses_2170762o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€F: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€F
 
_user_specified_nameinputs
©	
є
%__inference_signature_wrapper_2171127
hidden_input
unknown:	Fи
	unknown_0:	и
	unknown_1:	и
	unknown_2:	и
	unknown_3:	и
	unknown_4:	и
	unknown_5:	и$
	unknown_6:$
identityИҐStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallhidden_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_2170619o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€F: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:€€€€€€€€€F
&
_user_specified_namehidden_input
ы
Ш
J__inference_sequential_18_layer_call_and_return_conditional_losses_2170935
hidden_input!
hidden_2170914:	Fи
hidden_2170916:	и-
batch_normalization_18_2170920:	и-
batch_normalization_18_2170922:	и-
batch_normalization_18_2170924:	и-
batch_normalization_18_2170926:	и
out_2170929:	и$
out_2170931:$
identityИҐ.batch_normalization_18/StatefulPartitionedCallҐhidden/StatefulPartitionedCallҐout/StatefulPartitionedCallт
hidden/StatefulPartitionedCallStatefulPartitionedCallhidden_inputhidden_2170914hidden_2170916*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_layer_call_and_return_conditional_losses_2170722я
dropout_18/PartitionedCallPartitionedCall'hidden/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_18_layer_call_and_return_conditional_losses_2170733Н
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0batch_normalization_18_2170920batch_normalization_18_2170922batch_normalization_18_2170924batch_normalization_18_2170926*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2170643Р
out/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0out_2170929out_2170931*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_out_layer_call_and_return_conditional_losses_2170755s
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$ґ
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€F: : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:U Q
'
_output_shapes
:€€€€€€€€€F
&
_user_specified_namehidden_input
ƒ
Ч
(__inference_hidden_layer_call_fn_2171136

inputs
unknown:	Fи
	unknown_0:	и
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€и*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_layer_call_and_return_conditional_losses_2170722p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€и`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€F: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€F
 
_user_specified_nameinputs"џL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*∞
serving_defaultЬ
E
hidden_input5
serving_default_hidden_input:0€€€€€€€€€F7
out0
StatefulPartitionedCall:0€€€€€€€€€$tensorflow/serving/predict:Яf
и
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
ї

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Љ
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
к
axis
	gamma
beta
 moving_mean
!moving_variance
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
ї

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
њ
0iter

1beta_1

2beta_2
	3decay
4learning_ratemZm[m\m](m^)m_v`vavbvc(vd)ve"
	optimizer
X
0
1
2
3
 4
!5
(6
)7"
trackable_list_wrapper
J
0
1
2
3
(4
)5"
trackable_list_wrapper
 "
trackable_list_wrapper
 
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
К2З
/__inference_sequential_18_layer_call_fn_2170781
/__inference_sequential_18_layer_call_fn_2170986
/__inference_sequential_18_layer_call_fn_2171007
/__inference_sequential_18_layer_call_fn_2170911ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ц2у
J__inference_sequential_18_layer_call_and_return_conditional_losses_2171045
J__inference_sequential_18_layer_call_and_return_conditional_losses_2171104
J__inference_sequential_18_layer_call_and_return_conditional_losses_2170935
J__inference_sequential_18_layer_call_and_return_conditional_losses_2170959ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“Bѕ
"__inference__wrapped_model_2170619hidden_input"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
,
:serving_default"
signature_map
 :	Fи2hidden/kernel
:и2hidden/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
“2ѕ
(__inference_hidden_layer_call_fn_2171136Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_hidden_layer_call_and_return_conditional_losses_2171150Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ц2У
,__inference_dropout_18_layer_call_fn_2171155
,__inference_dropout_18_layer_call_fn_2171160і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ћ2…
G__inference_dropout_18_layer_call_and_return_conditional_losses_2171165
G__inference_dropout_18_layer_call_and_return_conditional_losses_2171177і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
 "
trackable_list_wrapper
+:)и2batch_normalization_18/gamma
*:(и2batch_normalization_18/beta
3:1и (2"batch_normalization_18/moving_mean
7:5и (2&batch_normalization_18/moving_variance
<
0
1
 2
!3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
Ѓ2Ђ
8__inference_batch_normalization_18_layer_call_fn_2171190
8__inference_batch_normalization_18_layer_call_fn_2171203і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2171223
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2171257і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
:	и$2
out/kernel
:$2out/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
ѕ2ћ
%__inference_out_layer_call_fn_2171266Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
@__inference_out_layer_call_and_return_conditional_losses_2171277Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
 0
!1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
—Bќ
%__inference_signature_wrapper_2171127hidden_input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
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
.
 0
!1"
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
N
	Qtotal
	Rcount
S	variables
T	keras_api"
_tf_keras_metric
^
	Utotal
	Vcount
W
_fn_kwargs
X	variables
Y	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
Q0
R1"
trackable_list_wrapper
-
S	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
U0
V1"
trackable_list_wrapper
-
X	variables"
_generic_user_object
%:#	Fи2Adam/hidden/kernel/m
:и2Adam/hidden/bias/m
0:.и2#Adam/batch_normalization_18/gamma/m
/:-и2"Adam/batch_normalization_18/beta/m
": 	и$2Adam/out/kernel/m
:$2Adam/out/bias/m
%:#	Fи2Adam/hidden/kernel/v
:и2Adam/hidden/bias/v
0:.и2#Adam/batch_normalization_18/gamma/v
/:-и2"Adam/batch_normalization_18/beta/v
": 	и$2Adam/out/kernel/v
:$2Adam/out/bias/vТ
"__inference__wrapped_model_2170619l! ()5Ґ2
+Ґ(
&К#
hidden_input€€€€€€€€€F
™ ")™&
$
outК
out€€€€€€€€€$ї
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2171223d! 4Ґ1
*Ґ'
!К
inputs€€€€€€€€€и
p 
™ "&Ґ#
К
0€€€€€€€€€и
Ъ ї
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2171257d !4Ґ1
*Ґ'
!К
inputs€€€€€€€€€и
p
™ "&Ґ#
К
0€€€€€€€€€и
Ъ У
8__inference_batch_normalization_18_layer_call_fn_2171190W! 4Ґ1
*Ґ'
!К
inputs€€€€€€€€€и
p 
™ "К€€€€€€€€€иУ
8__inference_batch_normalization_18_layer_call_fn_2171203W !4Ґ1
*Ґ'
!К
inputs€€€€€€€€€и
p
™ "К€€€€€€€€€и©
G__inference_dropout_18_layer_call_and_return_conditional_losses_2171165^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€и
p 
™ "&Ґ#
К
0€€€€€€€€€и
Ъ ©
G__inference_dropout_18_layer_call_and_return_conditional_losses_2171177^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€и
p
™ "&Ґ#
К
0€€€€€€€€€и
Ъ Б
,__inference_dropout_18_layer_call_fn_2171155Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€и
p 
™ "К€€€€€€€€€иБ
,__inference_dropout_18_layer_call_fn_2171160Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€и
p
™ "К€€€€€€€€€и§
C__inference_hidden_layer_call_and_return_conditional_losses_2171150]/Ґ,
%Ґ"
 К
inputs€€€€€€€€€F
™ "&Ґ#
К
0€€€€€€€€€и
Ъ |
(__inference_hidden_layer_call_fn_2171136P/Ґ,
%Ґ"
 К
inputs€€€€€€€€€F
™ "К€€€€€€€€€и°
@__inference_out_layer_call_and_return_conditional_losses_2171277]()0Ґ-
&Ґ#
!К
inputs€€€€€€€€€и
™ "%Ґ"
К
0€€€€€€€€€$
Ъ y
%__inference_out_layer_call_fn_2171266P()0Ґ-
&Ґ#
!К
inputs€€€€€€€€€и
™ "К€€€€€€€€€$Њ
J__inference_sequential_18_layer_call_and_return_conditional_losses_2170935p! ()=Ґ:
3Ґ0
&К#
hidden_input€€€€€€€€€F
p 

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ Њ
J__inference_sequential_18_layer_call_and_return_conditional_losses_2170959p !()=Ґ:
3Ґ0
&К#
hidden_input€€€€€€€€€F
p

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ Є
J__inference_sequential_18_layer_call_and_return_conditional_losses_2171045j! ()7Ґ4
-Ґ*
 К
inputs€€€€€€€€€F
p 

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ Є
J__inference_sequential_18_layer_call_and_return_conditional_losses_2171104j !()7Ґ4
-Ґ*
 К
inputs€€€€€€€€€F
p

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ Ц
/__inference_sequential_18_layer_call_fn_2170781c! ()=Ґ:
3Ґ0
&К#
hidden_input€€€€€€€€€F
p 

 
™ "К€€€€€€€€€$Ц
/__inference_sequential_18_layer_call_fn_2170911c !()=Ґ:
3Ґ0
&К#
hidden_input€€€€€€€€€F
p

 
™ "К€€€€€€€€€$Р
/__inference_sequential_18_layer_call_fn_2170986]! ()7Ґ4
-Ґ*
 К
inputs€€€€€€€€€F
p 

 
™ "К€€€€€€€€€$Р
/__inference_sequential_18_layer_call_fn_2171007] !()7Ґ4
-Ґ*
 К
inputs€€€€€€€€€F
p

 
™ "К€€€€€€€€€$•
%__inference_signature_wrapper_2171127|! ()EҐB
Ґ 
;™8
6
hidden_input&К#
hidden_input€€€€€€€€€F")™&
$
outК
out€€€€€€€€€$