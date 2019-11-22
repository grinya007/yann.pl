package AI::YANN::Layer;
use strict;
use warnings;

use AI::YANN::Parameter;
use AI::YANN::Utils qw/is_num/;
use Carp qw/confess/;

sub new {
  my ($class, $in_size, $out_size, $activation, $weight_sd) = @_;
  confess("Input size isn't numeric") unless (is_num($in_size));
  confess("Output size isn't numeric") unless (is_num($out_size));
  $activation //= 'linear';
  $weight_sd  //= 0.1;
  my $activation_f = "__fwd_$activation";
  my $activation_b = "__bwd_$activation";
  confess("Activation function $activation is not supported") unless (
    $class->can($activation_f)
  );
  return bless({
    '_W'              => AI::YANN::Parameter->new($in_size, $out_size, $weight_sd),
    '_b'              => AI::YANN::Parameter->new(1, $out_size, $weight_sd),
    '_activation_f'   => $activation_f,
    '_activation_b'   => $activation_b,
    '_input'          => undef,
    '_linear_out'     => undef,
    '_activation_out' => undef,
  }, $class);
}

sub update {
  my ($self, $lr) = @_;
  for my $p (qw/_W _b/) {
    my $g = $self->{$p}->gradient();
    $self->{$p}->add_momentum($g ** 2);
    $self->{$p}->add_value(
      (-$lr * $g)
      / ($self->{$p}->momentum() + 1e-8) ** 0.5
    );
  }
}

sub clean_up {
  my ($self) = @_;
  $self->{'_input'}           = undef;
  $self->{'_linear_out'}      = undef;
  $self->{'_activation_out'}  = undef;
}

sub zero_gradients {
  my ($self) = @_;
  $self->{'_W'}->set_gradient(undef);
  $self->{'_b'}->set_gradient(undef);
}

sub from_hash {
  my ($class, $hash) = @_;
  return $class->new(
    $hash->{'input_size'},
    $hash->{'output_size'},
    $hash->{'activation'},
    $hash->{'weight_sd'},
  );
}

sub forward {
  my ($self, $input) = @_;
  $self->{'_input'} = $input;
  $self->{'_linear_out'} = ($self->{'_W'}->value() x $input) + $self->{'_b'}->value();
  $self->{'_activation_out'} = $self->${ \$self->{'_activation_f'} }(
    $self->{'_linear_out'}
  );
  return $self->{'_activation_out'};
}

sub backward {
  my ($self, $d_activation_out_next) = @_;
  my $d_activation_out = $self->${ \$self->{'_activation_b'} }(
    $d_activation_out_next, $self->{'_linear_out'}
  );
  my $m = $self->{'_input'}->dim(0);
  $self->{'_W'}->set_gradient( (($d_activation_out x $self->{'_input'}->transpose()) / $m) );
  $self->{'_b'}->set_gradient( ($d_activation_out->sumover()->transpose() / $m) );

  return $self->{'_W'}->value()->transpose() x $d_activation_out;
}

sub __fwd_linear {
  my ($self, $x) = @_;
  return $x;
}

sub __bwd_linear {
  my ($self, $dx_next, $x) = @_;
  return $dx_next;
}

sub __fwd_relu {
  my ($self, $x) = @_;
  $x = $x->copy();
  $x->where($x < 0) .= 0;
  return $x;
}

sub __bwd_relu {
  my ($self, $dx_next, $x) = @_;
  $dx_next = $dx_next->copy();
  $dx_next->where($x < 0) .= 0;
  return $dx_next;
}

sub __fwd_sigmoid {
  my ($self, $x) = @_;
  return 1/(1 + exp(1) ** (-$x));
}

sub __bwd_sigmoid {
  my ($self, $dx_next, $x) = @_;
  my $sig = $self->__fwd_sigmoid($x);
  return $dx_next * $sig * (1 - $sig);
}

sub __fwd_softmax {
  my ($self, $x) = @_;
  my $exp = exp(1) ** ($x - $x->max());
  return $exp / $exp->transpose()->sumover();
}

sub __bwd_softmax {
  my ($self, $dx_next, $x) = @_;
  return $dx_next;
}

1;
