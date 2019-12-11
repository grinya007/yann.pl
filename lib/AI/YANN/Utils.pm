package AI::YANN::Utils;
use strict;
use warnings;

use Carp qw/confess/;
use Data::MessagePack;
use PDL::IO::FlexRaw qw/readflex writeflex/;
use Scalar::Util qw/blessed/;

use base 'Exporter';
our @EXPORT = qw//; 
our @EXPORT_OK = qw//;
our %EXPORT_TAGS = (
  'DATA'    => [qw/
    instance_of
    is_int
    is_num
    pdl_freeze
    pdl_thaw
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

sub instance_of {
  my ($var, $class) = @_;
  return blessed($var) && $var->isa($class);
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

my $_mp;
sub _mp {
  unless ($_mp) {
    $_mp = Data::MessagePack->new();
    $_mp->prefer_integer();
  }
  return $_mp;
}

sub pdl_freeze {
  my ($pdl) = @_;
  confess('Bad PDL object') unless (
    instance_of($pdl, 'PDL')
  );

  open(my $sfh, '>', \my $pdl_raw);
  my $hdr = writeflex($sfh, $pdl);
  close($sfh);

  return pack('(L/a)2', $pdl_raw, _mp()->pack($hdr));
}

sub pdl_thaw {
  my ($data) = @_;
  use bytes;
  confess('Bad blob') unless (
    defined($data) && !ref($data) && length($data) > 8
  );

  my ($pdl_raw, $hdr_raw) = unpack('(L/a)2', $data);
  open(my $pdl_sfh, '<', \$pdl_raw);

  return readflex($pdl_sfh, _mp()->unpack($hdr_raw));
}


1;
