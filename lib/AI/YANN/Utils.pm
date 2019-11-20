package AI::YANN::Utils;
use strict;
use warnings;

use base 'Exporter';
our @EXPORT = qw//; 
our @EXPORT_OK = qw//;
our %EXPORT_TAGS = (
  'DATA'    => [qw/
    is_int
    is_num
  /],
);

{
  my %export;
  @export{ map { @$_ } values %EXPORT_TAGS } = ();
  push @EXPORT_OK, keys %export;
  $EXPORT_TAGS{'ALL'} = \@EXPORT_OK;
}

sub is_int {
  my ($int) = @_;
  return undef unless (defined($int));
  no warnings 'numeric';
  return undef unless ($int eq int($int));
  return 1;
}

sub is_num {
  my ($num) = @_;
  return undef unless (defined($num));
  no warnings 'numeric';
  return undef unless ($num eq ($num + 0));
  return 1;
}

1;
