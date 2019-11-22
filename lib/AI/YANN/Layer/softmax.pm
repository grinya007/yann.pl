package AI::YANN::Layer::softmax;
use strict;
use warnings;

use base 'AI::YANN::Layer';

sub _forward {
  my ($self, $x) = @_;
  my $exp = exp(1) ** ($x - $x->max());
  return $exp / $exp->transpose()->sumover();
}

sub _backward {
  my ($self, $dx_next, $x) = @_;
  return $dx_next;
}

1;
