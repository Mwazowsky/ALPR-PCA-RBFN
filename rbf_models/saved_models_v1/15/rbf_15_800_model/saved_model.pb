ї№
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
 И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ег
w
hidden/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	†*
shared_namehidden/kernel
p
!hidden/kernel/Read/ReadVariableOpReadVariableOphidden/kernel*
_output_shapes
:	†*
dtype0
o
hidden/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:†*
shared_namehidden/bias
h
hidden/bias/Read/ReadVariableOpReadVariableOphidden/bias*
_output_shapes	
:†*
dtype0
П
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:†*,
shared_namebatch_normalization_1/gamma
И
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:†*
dtype0
Н
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:†*+
shared_namebatch_normalization_1/beta
Ж
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:†*
dtype0
Ы
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:†*2
shared_name#!batch_normalization_1/moving_mean
Ф
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:†*
dtype0
£
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:†*6
shared_name'%batch_normalization_1/moving_variance
Ь
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:†*
dtype0
q

out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	†$*
shared_name
out/kernel
j
out/kernel/Read/ReadVariableOpReadVariableOp
out/kernel*
_output_shapes
:	†$*
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
shape:	†*%
shared_nameAdam/hidden/kernel/m
~
(Adam/hidden/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden/kernel/m*
_output_shapes
:	†*
dtype0
}
Adam/hidden/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:†*#
shared_nameAdam/hidden/bias/m
v
&Adam/hidden/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden/bias/m*
_output_shapes	
:†*
dtype0
Э
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:†*3
shared_name$"Adam/batch_normalization_1/gamma/m
Ц
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes	
:†*
dtype0
Ы
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:†*2
shared_name#!Adam/batch_normalization_1/beta/m
Ф
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes	
:†*
dtype0

Adam/out/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	†$*"
shared_nameAdam/out/kernel/m
x
%Adam/out/kernel/m/Read/ReadVariableOpReadVariableOpAdam/out/kernel/m*
_output_shapes
:	†$*
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
shape:	†*%
shared_nameAdam/hidden/kernel/v
~
(Adam/hidden/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden/kernel/v*
_output_shapes
:	†*
dtype0
}
Adam/hidden/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:†*#
shared_nameAdam/hidden/bias/v
v
&Adam/hidden/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden/bias/v*
_output_shapes	
:†*
dtype0
Э
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:†*3
shared_name$"Adam/batch_normalization_1/gamma/v
Ц
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes	
:†*
dtype0
Ы
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:†*2
shared_name#!Adam/batch_normalization_1/beta/v
Ф
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes	
:†*
dtype0

Adam/out/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	†$*"
shared_nameAdam/out/kernel/v
x
%Adam/out/kernel/v/Read/ReadVariableOpReadVariableOpAdam/out/kernel/v*
_output_shapes
:	†$*
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
С4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ћ3
value¬3Bњ3 BЄ3
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
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
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
ОЗ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/out/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/out/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/hidden/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/hidden/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ОЗ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/out/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/out/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_hidden_inputPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
ь
StatefulPartitionedCallStatefulPartitionedCallserving_default_hidden_inputhidden/kernelhidden/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/beta
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
GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_126213
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ј
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!hidden/kernel/Read/ReadVariableOphidden/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOpout/kernel/Read/ReadVariableOpout/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/hidden/kernel/m/Read/ReadVariableOp&Adam/hidden/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp%Adam/out/kernel/m/Read/ReadVariableOp#Adam/out/bias/m/Read/ReadVariableOp(Adam/hidden/kernel/v/Read/ReadVariableOp&Adam/hidden/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp%Adam/out/kernel/v/Read/ReadVariableOp#Adam/out/bias/v/Read/ReadVariableOpConst**
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
GPU 2J 8В *(
f#R!
__inference__traced_save_126473
ч
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden/kernelhidden/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance
out/kernelout/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/hidden/kernel/mAdam/hidden/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/out/kernel/mAdam/out/bias/mAdam/hidden/kernel/vAdam/hidden/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/out/kernel/vAdam/out/bias/v*)
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
GPU 2J 8В *+
f&R$
"__inference__traced_restore_126570Ќя
Яu
“
"__inference__traced_restore_126570
file_prefix1
assignvariableop_hidden_kernel:	†-
assignvariableop_1_hidden_bias:	†=
.assignvariableop_2_batch_normalization_1_gamma:	†<
-assignvariableop_3_batch_normalization_1_beta:	†C
4assignvariableop_4_batch_normalization_1_moving_mean:	†G
8assignvariableop_5_batch_normalization_1_moving_variance:	†0
assignvariableop_6_out_kernel:	†$)
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
(assignvariableop_17_adam_hidden_kernel_m:	†5
&assignvariableop_18_adam_hidden_bias_m:	†E
6assignvariableop_19_adam_batch_normalization_1_gamma_m:	†D
5assignvariableop_20_adam_batch_normalization_1_beta_m:	†8
%assignvariableop_21_adam_out_kernel_m:	†$1
#assignvariableop_22_adam_out_bias_m:$;
(assignvariableop_23_adam_hidden_kernel_v:	†5
&assignvariableop_24_adam_hidden_bias_v:	†E
6assignvariableop_25_adam_batch_normalization_1_gamma_v:	†D
5assignvariableop_26_adam_batch_normalization_1_beta_v:	†8
%assignvariableop_27_adam_out_kernel_v:	†$1
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
:Э
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_1_moving_varianceIdentity_5:output:0"/device:CPU:0*
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
:І
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_batch_normalization_1_gamma_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_20AssignVariableOp5assignvariableop_20_adam_batch_normalization_1_beta_mIdentity_20:output:0"/device:CPU:0*
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
:І
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_batch_normalization_1_gamma_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adam_batch_normalization_1_beta_vIdentity_26:output:0"/device:CPU:0*
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
Й
х
B__inference_hidden_layer_call_and_return_conditional_losses_125808

inputs1
matmul_readvariableop_resource:	†.
biasadd_readvariableop_resource:	†
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	†*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:†*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†U
SquareSquareBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€†J
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
:€€€€€€€€€†F
ExpExpmul:z:0*
T0*(
_output_shapes
:€€€€€€€€€†W
IdentityIdentityExp:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€†w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
’8
Ё
!__inference__wrapped_model_125705
hidden_inputE
2sequential_1_hidden_matmul_readvariableop_resource:	†B
3sequential_1_hidden_biasadd_readvariableop_resource:	†S
Dsequential_1_batch_normalization_1_batchnorm_readvariableop_resource:	†W
Hsequential_1_batch_normalization_1_batchnorm_mul_readvariableop_resource:	†U
Fsequential_1_batch_normalization_1_batchnorm_readvariableop_1_resource:	†U
Fsequential_1_batch_normalization_1_batchnorm_readvariableop_2_resource:	†B
/sequential_1_out_matmul_readvariableop_resource:	†$>
0sequential_1_out_biasadd_readvariableop_resource:$
identityИҐ;sequential_1/batch_normalization_1/batchnorm/ReadVariableOpҐ=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1Ґ=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2Ґ?sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOpҐ*sequential_1/hidden/BiasAdd/ReadVariableOpҐ)sequential_1/hidden/MatMul/ReadVariableOpҐ'sequential_1/out/BiasAdd/ReadVariableOpҐ&sequential_1/out/MatMul/ReadVariableOpЭ
)sequential_1/hidden/MatMul/ReadVariableOpReadVariableOp2sequential_1_hidden_matmul_readvariableop_resource*
_output_shapes
:	†*
dtype0Ш
sequential_1/hidden/MatMulMatMulhidden_input1sequential_1/hidden/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†Ы
*sequential_1/hidden/BiasAdd/ReadVariableOpReadVariableOp3sequential_1_hidden_biasadd_readvariableop_resource*
_output_shapes	
:†*
dtype0≥
sequential_1/hidden/BiasAddBiasAdd$sequential_1/hidden/MatMul:product:02sequential_1/hidden/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†}
sequential_1/hidden/SquareSquare$sequential_1/hidden/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€†^
sequential_1/hidden/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  АњХ
sequential_1/hidden/mulMul"sequential_1/hidden/mul/x:output:0sequential_1/hidden/Square:y:0*
T0*(
_output_shapes
:€€€€€€€€€†n
sequential_1/hidden/ExpExpsequential_1/hidden/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€†{
sequential_1/dropout_1/IdentityIdentitysequential_1/hidden/Exp:y:0*
T0*(
_output_shapes
:€€€€€€€€€†љ
;sequential_1/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpDsequential_1_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:†*
dtype0w
2sequential_1/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:б
0sequential_1/batch_normalization_1/batchnorm/addAddV2Csequential_1/batch_normalization_1/batchnorm/ReadVariableOp:value:0;sequential_1/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:†Ч
2sequential_1/batch_normalization_1/batchnorm/RsqrtRsqrt4sequential_1/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:†≈
?sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_1_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:†*
dtype0ё
0sequential_1/batch_normalization_1/batchnorm/mulMul6sequential_1/batch_normalization_1/batchnorm/Rsqrt:y:0Gsequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:†ћ
2sequential_1/batch_normalization_1/batchnorm/mul_1Mul(sequential_1/dropout_1/Identity:output:04sequential_1/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€†Ѕ
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpFsequential_1_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:†*
dtype0№
2sequential_1/batch_normalization_1/batchnorm/mul_2MulEsequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1:value:04sequential_1/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:†Ѕ
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpFsequential_1_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:†*
dtype0№
0sequential_1/batch_normalization_1/batchnorm/subSubEsequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2:value:06sequential_1/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:†№
2sequential_1/batch_normalization_1/batchnorm/add_1AddV26sequential_1/batch_normalization_1/batchnorm/mul_1:z:04sequential_1/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€†Ч
&sequential_1/out/MatMul/ReadVariableOpReadVariableOp/sequential_1_out_matmul_readvariableop_resource*
_output_shapes
:	†$*
dtype0ї
sequential_1/out/MatMulMatMul6sequential_1/batch_normalization_1/batchnorm/add_1:z:0.sequential_1/out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$Ф
'sequential_1/out/BiasAdd/ReadVariableOpReadVariableOp0sequential_1_out_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0©
sequential_1/out/BiasAddBiasAdd!sequential_1/out/MatMul:product:0/sequential_1/out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€$x
sequential_1/out/SoftmaxSoftmax!sequential_1/out/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€$q
IdentityIdentity"sequential_1/out/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$т
NoOpNoOp<^sequential_1/batch_normalization_1/batchnorm/ReadVariableOp>^sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1>^sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2@^sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp+^sequential_1/hidden/BiasAdd/ReadVariableOp*^sequential_1/hidden/MatMul/ReadVariableOp(^sequential_1/out/BiasAdd/ReadVariableOp'^sequential_1/out/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : : : : : 2z
;sequential_1/batch_normalization_1/batchnorm/ReadVariableOp;sequential_1/batch_normalization_1/batchnorm/ReadVariableOp2~
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_1=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_12~
=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_2=sequential_1/batch_normalization_1/batchnorm/ReadVariableOp_22В
?sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp?sequential_1/batch_normalization_1/batchnorm/mul/ReadVariableOp2X
*sequential_1/hidden/BiasAdd/ReadVariableOp*sequential_1/hidden/BiasAdd/ReadVariableOp2V
)sequential_1/hidden/MatMul/ReadVariableOp)sequential_1/hidden/MatMul/ReadVariableOp2R
'sequential_1/out/BiasAdd/ReadVariableOp'sequential_1/out/BiasAdd/ReadVariableOp2P
&sequential_1/out/MatMul/ReadVariableOp&sequential_1/out/MatMul/ReadVariableOp:U Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namehidden_input
а
і
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_125729

inputs0
!batchnorm_readvariableop_resource:	†4
%batchnorm_mul_readvariableop_resource:	†2
#batchnorm_readvariableop_1_resource:	†2
#batchnorm_readvariableop_2_resource:	†
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:†*
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
:†Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:†
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:†*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:†d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€†{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:†*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:†{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:†*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:†s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€†c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€†Ї
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€†: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
’	
Ѕ
-__inference_sequential_1_layer_call_fn_125997
hidden_input
unknown:	†
	unknown_0:	†
	unknown_1:	†
	unknown_2:	†
	unknown_3:	†
	unknown_4:	†
	unknown_5:	†$
	unknown_6:$
identityИҐStatefulPartitionedCallѓ
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
GPU 2J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_125957o
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
#:€€€€€€€€€: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namehidden_input
Я

с
?__inference_out_layer_call_and_return_conditional_losses_125841

inputs1
matmul_readvariableop_resource:	†$-
biasadd_readvariableop_resource:$
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	†$*
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
:€€€€€€€€€†: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
№
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_125819

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€†\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€†"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€†:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
І	
Є
$__inference_signature_wrapper_126213
hidden_input
unknown:	†
	unknown_0:	†
	unknown_1:	†
	unknown_2:	†
	unknown_3:	†
	unknown_4:	†
	unknown_5:	†$
	unknown_6:$
identityИҐStatefulPartitionedCallК
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
GPU 2J 8В **
f%R#
!__inference__wrapped_model_125705o
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
#:€€€€€€€€€: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namehidden_input
ѓ%
о
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_125776

inputs6
'assignmovingavg_readvariableop_resource:	†8
)assignmovingavg_1_readvariableop_resource:	†4
%batchnorm_mul_readvariableop_resource:	†0
!batchnorm_readvariableop_resource:	†
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	†*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	†И
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€†l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	†*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:†*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:†*
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
:†*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:†y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:†ђ
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
:†*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:†
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:†і
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
:†Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:†
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:†*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:†d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€†i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:†w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:†*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:†s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€†c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€†к
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€†: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
„	
Ѕ
-__inference_sequential_1_layer_call_fn_125867
hidden_input
unknown:	†
	unknown_0:	†
	unknown_1:	†
	unknown_2:	†
	unknown_3:	†
	unknown_4:	†
	unknown_5:	†$
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
:€€€€€€€€€$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_125848o
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
#:€€€€€€€€€: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namehidden_input
з
І
H__inference_sequential_1_layer_call_and_return_conditional_losses_125957

inputs 
hidden_125936:	†
hidden_125938:	†+
batch_normalization_1_125942:	†+
batch_normalization_1_125944:	†+
batch_normalization_1_125946:	†+
batch_normalization_1_125948:	†

out_125951:	†$

out_125953:$
identityИҐ-batch_normalization_1/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐhidden/StatefulPartitionedCallҐout/StatefulPartitionedCallй
hidden/StatefulPartitionedCallStatefulPartitionedCallinputshidden_125936hidden_125938*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_hidden_layer_call_and_return_conditional_losses_125808м
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_125897З
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0batch_normalization_1_125942batch_normalization_1_125944batch_normalization_1_125946batch_normalization_1_125948*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_125776М
out/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0
out_125951
out_125953*
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
GPU 2J 8В *H
fCRA
?__inference_out_layer_call_and_return_conditional_losses_125841s
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$ў
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
х
c
*__inference_dropout_1_layer_call_fn_126246

inputs
identityИҐStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_125897p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€†`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€†22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
£
F
*__inference_dropout_1_layer_call_fn_126241

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_125819a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€†"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€†:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
¬
Ц
'__inference_hidden_layer_call_fn_126222

inputs
unknown:	†
	unknown_0:	†
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_hidden_layer_call_and_return_conditional_losses_125808p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€†`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
∞
’
6__inference_batch_normalization_1_layer_call_fn_126276

inputs
unknown:	†
	unknown_0:	†
	unknown_1:	†
	unknown_2:	†
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_125729p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€†`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€†: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
Й
х
B__inference_hidden_layer_call_and_return_conditional_losses_126236

inputs1
matmul_readvariableop_resource:	†.
biasadd_readvariableop_resource:	†
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	†*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:†*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†U
SquareSquareBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€†J
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
:€€€€€€€€€†F
ExpExpmul:z:0*
T0*(
_output_shapes
:€€€€€€€€€†W
IdentityIdentityExp:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€†w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
щ
≠
H__inference_sequential_1_layer_call_and_return_conditional_losses_126045
hidden_input 
hidden_126024:	†
hidden_126026:	†+
batch_normalization_1_126030:	†+
batch_normalization_1_126032:	†+
batch_normalization_1_126034:	†+
batch_normalization_1_126036:	†

out_126039:	†$

out_126041:$
identityИҐ-batch_normalization_1/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐhidden/StatefulPartitionedCallҐout/StatefulPartitionedCallп
hidden/StatefulPartitionedCallStatefulPartitionedCallhidden_inputhidden_126024hidden_126026*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_hidden_layer_call_and_return_conditional_losses_125808м
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_125897З
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0batch_normalization_1_126030batch_normalization_1_126032batch_normalization_1_126034batch_normalization_1_126036*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_125776М
out/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0
out_126039
out_126041*
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
GPU 2J 8В *H
fCRA
?__inference_out_layer_call_and_return_conditional_losses_125841s
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$ў
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:U Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namehidden_input
а
і
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_126309

inputs0
!batchnorm_readvariableop_resource:	†4
%batchnorm_mul_readvariableop_resource:	†2
#batchnorm_readvariableop_1_resource:	†2
#batchnorm_readvariableop_2_resource:	†
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:†*
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
:†Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:†
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:†*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:†d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€†{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:†*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:†{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:†*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:†s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€†c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€†Ї
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€†: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
Ѓ
’
6__inference_batch_normalization_1_layer_call_fn_126289

inputs
unknown:	†
	unknown_0:	†
	unknown_1:	†
	unknown_2:	†
identityИҐStatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_125776p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€†`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€†: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
ї
Т
$__inference_out_layer_call_fn_126352

inputs
unknown:	†$
	unknown_0:$
identityИҐStatefulPartitionedCall‘
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
GPU 2J 8В *H
fCRA
?__inference_out_layer_call_and_return_conditional_losses_125841o
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
:€€€€€€€€€†: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
√	
ї
-__inference_sequential_1_layer_call_fn_126093

inputs
unknown:	†
	unknown_0:	†
	unknown_1:	†
	unknown_2:	†
	unknown_3:	†
	unknown_4:	†
	unknown_5:	†$
	unknown_6:$
identityИҐStatefulPartitionedCall©
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
GPU 2J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_125957o
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
#:€€€€€€€€€: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Я

с
?__inference_out_layer_call_and_return_conditional_losses_126363

inputs1
matmul_readvariableop_resource:	†$-
biasadd_readvariableop_resource:$
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	†$*
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
:€€€€€€€€€†: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
ѓ%
о
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_126343

inputs6
'assignmovingavg_readvariableop_resource:	†8
)assignmovingavg_1_readvariableop_resource:	†4
%batchnorm_mul_readvariableop_resource:	†0
!batchnorm_readvariableop_resource:	†
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	†*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	†И
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€†l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	†*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:†*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:†*
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
:†*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:†y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:†ђ
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
:†*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:†
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:†і
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
:†Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:†
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:†*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:†d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€†i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:†w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:†*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:†s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€†c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€†к
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€†: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
ы	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_126263

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
:€€€€€€€€€†C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€†*
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
:€€€€€€€€€†p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€†j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€†Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€†"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€†:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
€?
≥
__inference__traced_save_126473
file_prefix,
(savev2_hidden_kernel_read_readvariableop*
&savev2_hidden_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop)
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
-savev2_adam_hidden_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop0
,savev2_adam_out_kernel_m_read_readvariableop.
*savev2_adam_out_bias_m_read_readvariableop3
/savev2_adam_hidden_kernel_v_read_readvariableop1
-savev2_adam_hidden_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop0
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
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B †
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_hidden_kernel_read_readvariableop&savev2_hidden_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop%savev2_out_kernel_read_readvariableop#savev2_out_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_hidden_kernel_m_read_readvariableop-savev2_adam_hidden_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop,savev2_adam_out_kernel_m_read_readvariableop*savev2_adam_out_bias_m_read_readvariableop/savev2_adam_hidden_kernel_v_read_readvariableop-savev2_adam_hidden_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop,savev2_adam_out_kernel_v_read_readvariableop*savev2_adam_out_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
Ј: :	†:†:†:†:†:†:	†$:$: : : : : : : : : :	†:†:†:†:	†$:$:	†:†:†:†:	†$:$: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	†:!

_output_shapes	
:†:!

_output_shapes	
:†:!

_output_shapes	
:†:!

_output_shapes	
:†:!

_output_shapes	
:†:%!

_output_shapes
:	†$: 
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
:	†:!

_output_shapes	
:†:!

_output_shapes	
:†:!

_output_shapes	
:†:%!

_output_shapes
:	†$: 

_output_shapes
:$:%!

_output_shapes
:	†:!

_output_shapes	
:†:!

_output_shapes	
:†:!

_output_shapes	
:†:%!

_output_shapes
:	†$: 

_output_shapes
:$:

_output_shapes
: 
№
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_126251

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€†\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€†"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€†:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
ўO
Ф
H__inference_sequential_1_layer_call_and_return_conditional_losses_126190

inputs8
%hidden_matmul_readvariableop_resource:	†5
&hidden_biasadd_readvariableop_resource:	†L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	†N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	†J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	†F
7batch_normalization_1_batchnorm_readvariableop_resource:	†5
"out_matmul_readvariableop_resource:	†$1
#out_biasadd_readvariableop_resource:$
identityИҐ%batch_normalization_1/AssignMovingAvgҐ4batch_normalization_1/AssignMovingAvg/ReadVariableOpҐ'batch_normalization_1/AssignMovingAvg_1Ґ6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_1/batchnorm/ReadVariableOpҐ2batch_normalization_1/batchnorm/mul/ReadVariableOpҐhidden/BiasAdd/ReadVariableOpҐhidden/MatMul/ReadVariableOpҐout/BiasAdd/ReadVariableOpҐout/MatMul/ReadVariableOpГ
hidden/MatMul/ReadVariableOpReadVariableOp%hidden_matmul_readvariableop_resource*
_output_shapes
:	†*
dtype0x
hidden/MatMulMatMulinputs$hidden/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†Б
hidden/BiasAdd/ReadVariableOpReadVariableOp&hidden_biasadd_readvariableop_resource*
_output_shapes	
:†*
dtype0М
hidden/BiasAddBiasAddhidden/MatMul:product:0%hidden/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†c
hidden/SquareSquarehidden/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€†Q
hidden/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  Ањn

hidden/mulMulhidden/mul/x:output:0hidden/Square:y:0*
T0*(
_output_shapes
:€€€€€€€€€†T

hidden/ExpExphidden/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€†\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Б
dropout_1/dropout/MulMulhidden/Exp:y:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€†U
dropout_1/dropout/ShapeShapehidden/Exp:y:0*
T0*
_output_shapes
:°
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€†*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?≈
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€†Д
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€†И
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€†~
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ѕ
"batch_normalization_1/moments/meanMeandropout_1/dropout/Mul_1:z:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	†*
	keep_dims(С
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	†…
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedropout_1/dropout/Mul_1:z:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€†В
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: б
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	†*
	keep_dims(Ъ
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:†*
squeeze_dims
 †
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:†*
squeeze_dims
 p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<ѓ
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:†*
dtype0ƒ
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:†ї
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:†Д
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<≥
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:†*
dtype0 
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:†Ѕ
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:†М
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:і
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:†}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:†Ђ
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:†*
dtype0Ј
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:†•
%batch_normalization_1/batchnorm/mul_1Muldropout_1/dropout/Mul_1:z:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€†Ђ
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:†£
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:†*
dtype0≥
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:†µ
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€†}
out/MatMul/ReadVariableOpReadVariableOp"out_matmul_readvariableop_resource*
_output_shapes
:	†$*
dtype0Ф

out/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0!out/MatMul/ReadVariableOp:value:0*
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
:€€€€€€€€€$ж
NoOpNoOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp^hidden/BiasAdd/ReadVariableOp^hidden/MatMul/ReadVariableOp^out/BiasAdd/ReadVariableOp^out/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : : : : : 2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2>
hidden/BiasAdd/ReadVariableOphidden/BiasAdd/ReadVariableOp2<
hidden/MatMul/ReadVariableOphidden/MatMul/ReadVariableOp28
out/BiasAdd/ReadVariableOpout/BiasAdd/ReadVariableOp26
out/MatMul/ReadVariableOpout/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ы	
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_125897

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
:€€€€€€€€€†C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€†*
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
:€€€€€€€€€†p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€†j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€†Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€†"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€†:P L
(
_output_shapes
:€€€€€€€€€†
 
_user_specified_nameinputs
√.
Ѓ
H__inference_sequential_1_layer_call_and_return_conditional_losses_126131

inputs8
%hidden_matmul_readvariableop_resource:	†5
&hidden_biasadd_readvariableop_resource:	†F
7batch_normalization_1_batchnorm_readvariableop_resource:	†J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	†H
9batch_normalization_1_batchnorm_readvariableop_1_resource:	†H
9batch_normalization_1_batchnorm_readvariableop_2_resource:	†5
"out_matmul_readvariableop_resource:	†$1
#out_biasadd_readvariableop_resource:$
identityИҐ.batch_normalization_1/batchnorm/ReadVariableOpҐ0batch_normalization_1/batchnorm/ReadVariableOp_1Ґ0batch_normalization_1/batchnorm/ReadVariableOp_2Ґ2batch_normalization_1/batchnorm/mul/ReadVariableOpҐhidden/BiasAdd/ReadVariableOpҐhidden/MatMul/ReadVariableOpҐout/BiasAdd/ReadVariableOpҐout/MatMul/ReadVariableOpГ
hidden/MatMul/ReadVariableOpReadVariableOp%hidden_matmul_readvariableop_resource*
_output_shapes
:	†*
dtype0x
hidden/MatMulMatMulinputs$hidden/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†Б
hidden/BiasAdd/ReadVariableOpReadVariableOp&hidden_biasadd_readvariableop_resource*
_output_shapes	
:†*
dtype0М
hidden/BiasAddBiasAddhidden/MatMul:product:0%hidden/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€†c
hidden/SquareSquarehidden/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€†Q
hidden/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  Ањn

hidden/mulMulhidden/mul/x:output:0hidden/Square:y:0*
T0*(
_output_shapes
:€€€€€€€€€†T

hidden/ExpExphidden/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€†a
dropout_1/IdentityIdentityhidden/Exp:y:0*
T0*(
_output_shapes
:€€€€€€€€€†£
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:†*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ї
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:†}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:†Ђ
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:†*
dtype0Ј
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:†•
%batch_normalization_1/batchnorm/mul_1Muldropout_1/Identity:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€†І
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:†*
dtype0µ
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:†І
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:†*
dtype0µ
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:†µ
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€†}
out/MatMul/ReadVariableOpReadVariableOp"out_matmul_readvariableop_resource*
_output_shapes
:	†$*
dtype0Ф

out/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0!out/MatMul/ReadVariableOp:value:0*
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
:€€€€€€€€€$К
NoOpNoOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp^hidden/BiasAdd/ReadVariableOp^hidden/MatMul/ReadVariableOp^out/BiasAdd/ReadVariableOp^out/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : : : : : 2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2>
hidden/BiasAdd/ReadVariableOphidden/BiasAdd/ReadVariableOp2<
hidden/MatMul/ReadVariableOphidden/MatMul/ReadVariableOp28
out/BiasAdd/ReadVariableOpout/BiasAdd/ReadVariableOp26
out/MatMul/ReadVariableOpout/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
”
Й
H__inference_sequential_1_layer_call_and_return_conditional_losses_126021
hidden_input 
hidden_126000:	†
hidden_126002:	†+
batch_normalization_1_126006:	†+
batch_normalization_1_126008:	†+
batch_normalization_1_126010:	†+
batch_normalization_1_126012:	†

out_126015:	†$

out_126017:$
identityИҐ-batch_normalization_1/StatefulPartitionedCallҐhidden/StatefulPartitionedCallҐout/StatefulPartitionedCallп
hidden/StatefulPartitionedCallStatefulPartitionedCallhidden_inputhidden_126000hidden_126002*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_hidden_layer_call_and_return_conditional_losses_125808№
dropout_1/PartitionedCallPartitionedCall'hidden/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_125819Б
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0batch_normalization_1_126006batch_normalization_1_126008batch_normalization_1_126010batch_normalization_1_126012*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_125729М
out/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0
out_126015
out_126017*
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
GPU 2J 8В *H
fCRA
?__inference_out_layer_call_and_return_conditional_losses_125841s
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$µ
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:U Q
'
_output_shapes
:€€€€€€€€€
&
_user_specified_namehidden_input
≈	
ї
-__inference_sequential_1_layer_call_fn_126072

inputs
unknown:	†
	unknown_0:	†
	unknown_1:	†
	unknown_2:	†
	unknown_3:	†
	unknown_4:	†
	unknown_5:	†$
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
:€€€€€€€€€$**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_125848o
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
#:€€€€€€€€€: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѕ
Г
H__inference_sequential_1_layer_call_and_return_conditional_losses_125848

inputs 
hidden_125809:	†
hidden_125811:	†+
batch_normalization_1_125821:	†+
batch_normalization_1_125823:	†+
batch_normalization_1_125825:	†+
batch_normalization_1_125827:	†

out_125842:	†$

out_125844:$
identityИҐ-batch_normalization_1/StatefulPartitionedCallҐhidden/StatefulPartitionedCallҐout/StatefulPartitionedCallй
hidden/StatefulPartitionedCallStatefulPartitionedCallinputshidden_125809hidden_125811*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_hidden_layer_call_and_return_conditional_losses_125808№
dropout_1/PartitionedCallPartitionedCall'hidden/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_125819Б
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0batch_normalization_1_125821batch_normalization_1_125823batch_normalization_1_125825batch_normalization_1_125827*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€†*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_125729М
out/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0
out_125842
out_125844*
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
GPU 2J 8В *H
fCRA
?__inference_out_layer_call_and_return_conditional_losses_125841s
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€$µ
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€: : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
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
serving_default_hidden_input:0€€€€€€€€€7
out0
StatefulPartitionedCall:0€€€€€€€€€$tensorflow/serving/predict:…e
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
В2€
-__inference_sequential_1_layer_call_fn_125867
-__inference_sequential_1_layer_call_fn_126072
-__inference_sequential_1_layer_call_fn_126093
-__inference_sequential_1_layer_call_fn_125997ј
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
о2л
H__inference_sequential_1_layer_call_and_return_conditional_losses_126131
H__inference_sequential_1_layer_call_and_return_conditional_losses_126190
H__inference_sequential_1_layer_call_and_return_conditional_losses_126021
H__inference_sequential_1_layer_call_and_return_conditional_losses_126045ј
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
—Bќ
!__inference__wrapped_model_125705hidden_input"Ш
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
 :	†2hidden/kernel
:†2hidden/bias
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
—2ќ
'__inference_hidden_layer_call_fn_126222Ґ
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
м2й
B__inference_hidden_layer_call_and_return_conditional_losses_126236Ґ
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
Т2П
*__inference_dropout_1_layer_call_fn_126241
*__inference_dropout_1_layer_call_fn_126246і
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
»2≈
E__inference_dropout_1_layer_call_and_return_conditional_losses_126251
E__inference_dropout_1_layer_call_and_return_conditional_losses_126263і
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
*:(†2batch_normalization_1/gamma
):'†2batch_normalization_1/beta
2:0† (2!batch_normalization_1/moving_mean
6:4† (2%batch_normalization_1/moving_variance
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
™2І
6__inference_batch_normalization_1_layer_call_fn_126276
6__inference_batch_normalization_1_layer_call_fn_126289і
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
а2Ё
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_126309
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_126343і
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
:	†$2
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
ќ2Ћ
$__inference_out_layer_call_fn_126352Ґ
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
й2ж
?__inference_out_layer_call_and_return_conditional_losses_126363Ґ
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
–BЌ
$__inference_signature_wrapper_126213hidden_input"Ф
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
%:#	†2Adam/hidden/kernel/m
:†2Adam/hidden/bias/m
/:-†2"Adam/batch_normalization_1/gamma/m
.:,†2!Adam/batch_normalization_1/beta/m
": 	†$2Adam/out/kernel/m
:$2Adam/out/bias/m
%:#	†2Adam/hidden/kernel/v
:†2Adam/hidden/bias/v
/:-†2"Adam/batch_normalization_1/gamma/v
.:,†2!Adam/batch_normalization_1/beta/v
": 	†$2Adam/out/kernel/v
:$2Adam/out/bias/vС
!__inference__wrapped_model_125705l! ()5Ґ2
+Ґ(
&К#
hidden_input€€€€€€€€€
™ ")™&
$
outК
out€€€€€€€€€$є
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_126309d! 4Ґ1
*Ґ'
!К
inputs€€€€€€€€€†
p 
™ "&Ґ#
К
0€€€€€€€€€†
Ъ є
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_126343d !4Ґ1
*Ґ'
!К
inputs€€€€€€€€€†
p
™ "&Ґ#
К
0€€€€€€€€€†
Ъ С
6__inference_batch_normalization_1_layer_call_fn_126276W! 4Ґ1
*Ґ'
!К
inputs€€€€€€€€€†
p 
™ "К€€€€€€€€€†С
6__inference_batch_normalization_1_layer_call_fn_126289W !4Ґ1
*Ґ'
!К
inputs€€€€€€€€€†
p
™ "К€€€€€€€€€†І
E__inference_dropout_1_layer_call_and_return_conditional_losses_126251^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€†
p 
™ "&Ґ#
К
0€€€€€€€€€†
Ъ І
E__inference_dropout_1_layer_call_and_return_conditional_losses_126263^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€†
p
™ "&Ґ#
К
0€€€€€€€€€†
Ъ 
*__inference_dropout_1_layer_call_fn_126241Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€†
p 
™ "К€€€€€€€€€†
*__inference_dropout_1_layer_call_fn_126246Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€†
p
™ "К€€€€€€€€€†£
B__inference_hidden_layer_call_and_return_conditional_losses_126236]/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "&Ґ#
К
0€€€€€€€€€†
Ъ {
'__inference_hidden_layer_call_fn_126222P/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€††
?__inference_out_layer_call_and_return_conditional_losses_126363]()0Ґ-
&Ґ#
!К
inputs€€€€€€€€€†
™ "%Ґ"
К
0€€€€€€€€€$
Ъ x
$__inference_out_layer_call_fn_126352P()0Ґ-
&Ґ#
!К
inputs€€€€€€€€€†
™ "К€€€€€€€€€$Љ
H__inference_sequential_1_layer_call_and_return_conditional_losses_126021p! ()=Ґ:
3Ґ0
&К#
hidden_input€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ Љ
H__inference_sequential_1_layer_call_and_return_conditional_losses_126045p !()=Ґ:
3Ґ0
&К#
hidden_input€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ ґ
H__inference_sequential_1_layer_call_and_return_conditional_losses_126131j! ()7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ ґ
H__inference_sequential_1_layer_call_and_return_conditional_losses_126190j !()7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€$
Ъ Ф
-__inference_sequential_1_layer_call_fn_125867c! ()=Ґ:
3Ґ0
&К#
hidden_input€€€€€€€€€
p 

 
™ "К€€€€€€€€€$Ф
-__inference_sequential_1_layer_call_fn_125997c !()=Ґ:
3Ґ0
&К#
hidden_input€€€€€€€€€
p

 
™ "К€€€€€€€€€$О
-__inference_sequential_1_layer_call_fn_126072]! ()7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€$О
-__inference_sequential_1_layer_call_fn_126093] !()7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€$§
$__inference_signature_wrapper_126213|! ()EҐB
Ґ 
;™8
6
hidden_input&К#
hidden_input€€€€€€€€€")™&
$
outК
out€€€€€€€€€$