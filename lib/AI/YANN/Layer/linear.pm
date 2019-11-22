package AI::YANN::Layer::linear;
use strict;
use warnings;

use base 'AI::YANN::Layer';

sub _forward {
  my ($self, $x) = @_;
  return $x;
}

sub _backward {
  my ($self, $dx_next, $x) = @_;
  return $dx_next;
}

1;
