package AI::YANN::Layer::relu;
use strict;
use warnings;

use base 'AI::YANN::Layer';

sub _forward {
  my ($self, $x) = @_;
  $x = $x->copy();
  $x->where($x < 0) .= 0;
  return $x;
}

sub _backward {
  my ($self, $dx_next, $x) = @_;
  $dx_next = $dx_next->copy();
  $dx_next->where($x < 0) .= 0;
  return $dx_next;
}

1;
