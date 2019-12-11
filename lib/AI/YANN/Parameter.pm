package AI::YANN::Parameter;
use strict;
use warnings;

use AI::YANN::Utils qw/pdl_freeze pdl_thaw/;
use PDL;

sub new {
  my ($class, $m, $n, $sd, $value, $momentum) = @_;
  return bless({
    '_value'    => $value // grandom($m, $n) * $sd,
    '_momentum' => $momentum // 0,
    '_gradient' => undef,
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

sub freeze {
  my ($self) = @_;
  return pack(
    '(L/a)2',
    pdl_freeze($self->{'_value'}),
    pdl_freeze($self->{'_momentum'}),
  );
}

sub thaw {
  my ($class, $blob) = @_;
  my ($value, $momentum) = unpack('(L/a)2', $blob);
  $value = pdl_thaw($value);
  $momentum = pdl_thaw($momentum);
  return $class->new(
    undef, undef, undef, $value, $momentum
  );
}

1;
