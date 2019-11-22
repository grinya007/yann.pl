package AI::YANN::Utils;
use strict;
use warnings;

use Carp qw/confess/;

use base 'Exporter';
our @EXPORT = qw//; 
our @EXPORT_OK = qw//;
our %EXPORT_TAGS = (
  'DATA'    => [qw/
    is_int
    is_num
  /],
  'SYSTEM'  => [qw/
    runtime_require
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

sub runtime_require {
  my ($pkg_name, %opts) = @_;
  confess(
    "invalid package name: '$pkg_name'"
  ) if ($pkg_name !~ /^[a-z0-9:_\-]+$/i);
  my $file_name = $pkg_name;
  $file_name =~ s!::+!/!g;
  $file_name .= '.pm' if ($file_name !~ /\.pm$/s);
  if (! exists( $INC{$file_name} )) {
    my $orig_sigdie = $SIG{'__DIE__'};
    local $SIG{'__DIE__'} = 'IGNORE' if ($opts{'try'});
    eval('require ' . $pkg_name);
    my $ex = $@;
    $SIG{'__DIE__'} = $orig_sigdie;
    if ($ex) {
      delete($INC{$file_name});
      return undef if (
        $opts{'try'} &&
        $ex =~ /^Can't locate $file_name/
      );
      confess($ex);
    }
  }
  return $pkg_name;
}


1;
