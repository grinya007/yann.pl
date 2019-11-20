package AI::YANN::Parameter;
use strict;
use warnings;

use PDL;

sub new {
  my ($class, $m, $n, $sd) = @_;
  my $value     = random($m, $n) * $sd;
  return bless({
    '_value'    => $value,
    '_gradient' => undef,
    '_momentum' => undef,
  }, $class);
}

sub value {
  my ($self) = @_;
  return $self->{'_value'};
}

sub add_value {
  my ($self, $v) = @_;
  $self->{'_value'} += $v;
}

sub gradient {
  my ($self) = @_;
  return $self->{'_gradient'};
}

sub set_gradient {
  my ($self, $g) = @_;
  $self->{'_gradient'} = $g;
}

sub momentum {
  my ($self) = @_;
  return $self->{'_momentum'};
}

sub add_momentum {
  my ($self, $m) = @_;
  $self->{'_momentum'} += $m;
}

1;
