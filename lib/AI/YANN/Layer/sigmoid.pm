package AI::YANN::Layer::sigmoid;
use strict;
use warnings;

use base 'AI::YANN::Layer';

sub _forward {
  my ($self, $x) = @_;
  return 1/(1 + exp(1) ** (-$x));
}

sub _backward {
  my ($self, $dx_next, $x) = @_;
  my $sig = $self->_forward($x);
  return $dx_next * $sig * (1 - $sig);
}

1;
