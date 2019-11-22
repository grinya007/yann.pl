package AI::YANN::Layer;
use strict;
use warnings;

use AI::YANN::Parameter;
use AI::YANN::Utils qw/is_num runtime_require/;
use Carp qw/confess/;

sub new {
  my ($class, $in_size, $out_size, $activation, $weight_sd) = @_;
  confess("Input size isn't numeric") unless (is_num($in_size));
  confess("Output size isn't numeric") unless (is_num($out_size));
  $activation //= 'linear';
  $weight_sd  //= 0.1;

  return bless({
    '_W'              => AI::YANN::Parameter->new($in_size, $out_size, $weight_sd),
    '_b'              => AI::YANN::Parameter->new(1, $out_size, $weight_sd),
    '_input'          => undef,
    '_linear_out'     => undef,
    '_activation_out' => undef,
  }, runtime_require("${class}::$activation"));
}

sub parameters {
  my ($self) = @_;
  return [ @$self{qw/_W _b/} ];
}

sub clean_up {
  my ($self) = @_;
  $self->{'_input'}           = undef;
  $self->{'_linear_out'}      = undef;
  $self->{'_activation_out'}  = undef;
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
  $self->{'_activation_out'} = $self->_forward($self->{'_linear_out'});
  return $self->{'_activation_out'};
}

sub backward {
  my ($self, $d_activation_out_next) = @_;
  my $d_activation_out = $self->_backward($d_activation_out_next, $self->{'_linear_out'});
  
  my $m = $self->{'_input'}->dim(0);
  $self->{'_W'}->set_gradient( ($d_activation_out x $self->{'_input'}->transpose()) / $m );
  $self->{'_b'}->set_gradient( $d_activation_out->sumover()->transpose() / $m );

  return $self->{'_W'}->value()->transpose() x $d_activation_out;
}

# activation, d(activation)/d(linear_out)
sub _forward  { ... }
sub _backward { ... }


1;
