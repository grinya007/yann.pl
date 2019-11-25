package AI::YANN::Layer;
use strict;
use warnings;

use AI::YANN::Parameter;
use AI::YANN::Utils qw/is_num runtime_require/;
use Carp qw/confess/;

sub new {
  my ($class, $activation, $out_size, $in_size, $input_idx, $weight_sd, $input_layers) = @_;
  confess("Input size isn't numeric") unless (is_num($in_size));
  confess("Output size isn't numeric") unless (is_num($out_size));
  $activation   //= 'linear';
  $weight_sd    //= 0.1;

  return bless({
    '_W'              => AI::YANN::Parameter->new($in_size, $out_size, $weight_sd),
    '_b'              => AI::YANN::Parameter->new(1, $out_size, $weight_sd),
    '_input_idx'      => $input_idx,
    '_input_layers'   => $input_layers,
    '_input_size'     => $in_size,
    '_output_size'    => $out_size,
    '_input'          => undef,
    '_linear_out'     => undef,
    '_activation_out' => undef,
  }, runtime_require("${class}::$activation"));
}

sub builder {
  my ($class) = @_;
  return sub {
    my (@args) = @_;
    my @new_args = ((undef) x 6);
    for my $i (0 .. $#args) {
      if (ref $args[$i]) {
        $new_args[2] //= 0;
        $new_args[2] += $args[$i]->output_size(),
        $new_args[5] //= [];
        push(@{ $new_args[5] }, $args[$i])
      }
      else {
        $new_args[$i] = $args[$i]
      }
    }
    return $class->new(@new_args);
  };
}

sub input_layers {
  my ($self) = @_;
  return $self->{'_input_layers'};
}

sub output_size {
  my ($self) = @_;
  return $self->{'_output_size'};
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

sub forward {
  my ($self, $input) = @_;
  if (defined $self->{'_input_idx'}) {
    $self->{'_input'} = $input->[ $self->{'_input_idx'} ]->transpose();
  }
  else {
    $self->{'_input'} = $input;
  }
  $self->{'_linear_out'} = ($self->{'_W'}->value() x $self->{'_input'}) + $self->{'_b'}->value();
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

# activation
sub _forward  { ... }
# d(activation)/d(linear_out)
sub _backward { ... }


1;
