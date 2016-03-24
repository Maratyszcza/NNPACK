#pragma once

namespace samples {
	namespace fft2 {
		static const float input[4] = {
			0.74090862, 0.85228336,
			0.71020633, 0.69013137
		};
		static const float output[4] = {
			1.45111495, 1.54241472,
			0.03070229, 0.16215199
		};
	}	

	namespace fft4 {
		static const float input[8] = {
			0x1.A8E4F8p-2f, 0x1.EE376Cp-3f,
			0x1.E29E82p-1f, 0x1.BD43D2p-1f,
			0x1.3E6038p-3f, 0x1.CC237Ep-1f,
			0x1.1CB97Ep-2f, 0x1.ECC3C2p-2f
		};
		static const float output[8] = {
			+0x1.CA82E58p+0, +0x1.3ED5C30p+1,
			+0x1.4BBC5F0p-1, -0x1.526BB30p+0,
			-0x1.4CF0B70p-1, -0x1.AFD1680p-3,
			-0x1.081E0C0p-3, +0x1.D610000p-8
		};
	}

	namespace fft8 {
		static const float input[16] = {
			0x1.98523Cp-1f, 0x1.60DCEEp-1f,
			0x1.28DDCEp-1f, 0x1.3184C6p-3f,
			0x1.381A28p-2f, 0x1.89A9AEp-4f,
			0x1.86A262p-1f, 0x1.5C77B0p-2f,
			0x1.543A9Cp-3f, 0x1.6B6768p-4f,
			0x1.9F272Cp-2f, 0x1.D79460p-2f,
			0x1.8F4D8Ep-5f, 0x1.351D14p-4f,
			0x1.DF2BFAp-3f, 0x1.FF459Cp-5f
		};
		static const float output[16] = {
			+0x1.A65066p+1f, +0x1.F63F24p+0f,
			+0x1.82AFA2p-2f, -0x1.23A292p-1f,
			+0x1.A25A26p-1f, +0x1.3C6D00p-1f,
			+0x1.ACF550p-1f, +0x1.8D3520p-1f,
			-0x1.547BF4p-1f, -0x1.01C700p-4f,
			+0x1.DA5280p-1f, +0x1.42290Ep+0f,
			+0x1.9CC78Cp-2f, +0x1.307506p-1f,
			+0x1.88DD68p-2f, +0x1.DFDB58p-1f
		};
	}

	namespace fft16 {
		static const float input[32] = {
			0x1.030E1Ep-1f, 0x1.7469A4p-2f,
			0x1.D9ECC6p-1f, 0x1.A8F096p-2f,
			0x1.7E31C2p-1f, 0x1.A212FEp-2f,
			0x1.EA9412p-1f, 0x1.F59FD0p-2f,
			0x1.BA8B0Ap-1f, 0x1.AE7DB2p-1f,
			0x1.49DA2Cp-2f, 0x1.72D13Cp-3f,
			0x1.83F174p-4f, 0x1.815008p-3f,
			0x1.B3DA54p-1f, 0x1.A1E5F2p-1f,
			0x1.D6F63Ep-1f, 0x1.01CF28p-3f,
			0x1.D4AA36p-1f, 0x1.2F5ECEp-1f,
			0x1.B6A5BEp-2f, 0x1.240B70p-2f,
			0x1.7179E4p-1f, 0x1.7F19E2p-1f,
			0x1.CFE398p-1f, 0x1.92646Ep-3f,
			0x1.2BAEE6p-1f, 0x1.F6947Cp-2f,
			0x1.C5DD26p-1f, 0x1.05F6F6p-4f,
			0x1.C46E5Ap-1f, 0x1.A14BCCp-1f
		};
		static const float output[32] = {
			+0x1.707DC8p+3f, +0x1.C15A56p+2f,
			+0x1.A81C9Ap-1f, +0x1.00A50Cp-1f,
			+0x1.82B37Ep-1f, -0x1.B1C5F2p-1f,
			-0x1.078E8Cp+1f, +0x1.B467E4p-2f,
			-0x1.32CB66p-3f, +0x1.3F607Ep+0f,
			-0x1.D5B318p-3f, +0x1.249074p-4f,
			-0x1.86945Cp+0f, -0x1.7D90DEp+0f,
			-0x1.4571A4p+0f, +0x1.3F427Cp-5f,
			-0x1.9F36A8p-1f, -0x1.09C3ECp+1f,
			+0x1.8D4172p+0f, +0x1.73B30Cp-1f,
			-0x1.1F8F5Ep-1f, -0x1.45690Ap-1f,
			-0x1.46F9C8p+0f, +0x1.465D12p-1f,
			+0x1.1D763Ap+1f, -0x1.5C32D2p-4f,
			-0x1.3AA070p+0f, -0x1.720F38p-3f,
			-0x1.7A4812p-5f, +0x1.9097F2p-1f,
			+0x1.7FBC54p-2f, -0x1.45C688p-2f
		};
	}

	namespace fft32 {
		static const float input[64] = {
			0x1.D93A22p-2f, 0x1.1A07EAp-5f,
			0x1.457616p-2f, 0x1.1C6E3Cp-4f,
			0x1.C1B5A4p-2f, 0x1.63CBB2p-1f,
			0x1.B1DBC2p-1f, 0x1.103DE0p-1f,
			0x1.8749D0p-3f, 0x1.528BBEp-2f,
			0x1.5A66D6p-4f, 0x1.85E378p-1f,
			0x1.6B7BAAp-1f, 0x1.4BE042p-1f,
			0x1.416FEAp-1f, 0x1.185BD0p-4f,
			0x1.8DD5B2p-1f, 0x1.2EE168p-1f,
			0x1.4A4EE2p-1f, 0x1.ECE9C2p-1f,
			0x1.961888p-3f, 0x1.97A598p-1f,
			0x1.469200p-3f, 0x1.A7AE76p-1f,
			0x1.C39A76p-2f, 0x1.C4A146p-1f,
			0x1.9F43C0p-1f, 0x1.BF5834p-4f,
			0x1.22CD6Ep-1f, 0x1.DB698Cp-1f,
			0x1.288430p-1f, 0x1.B4FE36p-2f,
			0x1.429398p-3f, 0x1.B4CA7Ep-1f,
			0x1.225E00p-4f, 0x1.DF5ED4p-1f,
			0x1.FA788Ap-2f, 0x1.AAC868p-1f,
			0x1.EF25F6p-1f, 0x1.C80E82p-1f,
			0x1.B40D9Cp-1f, 0x1.13D60Ap-1f,
			0x1.90D116p-1f, 0x1.D01774p-1f,
			0x1.80924Cp-1f, 0x1.A5AE34p-2f,
			0x1.11A7DAp-1f, 0x1.112618p-1f,
			0x1.86EAFEp-2f, 0x1.17E440p-1f,
			0x1.26EF72p-2f, 0x1.722E82p-4f,
			0x1.E8E2C6p-3f, 0x1.2ED9A4p-4f,
			0x1.A783EAp-2f, 0x1.DCA8FAp-6f,
			0x1.12C424p-1f, 0x1.43644Ap-3f,
			0x1.F030F8p-2f, 0x1.96FA28p-1f,
			0x1.0F1AD8p-2f, 0x1.9F5720p-1f,
			0x1.72F744p-1f, 0x1.911D50p-1f
		};
		static const float output[64] = {
			+0x1.F9A804p+3f, +0x1.1DB7F2p+4f,
			+0x1.672C94p-1f, -0x1.4A3D9Cp+1f,
			+0x1.B27418p-2f, -0x1.B14E6Cp-3f,
			-0x1.ABAF6Ap-2f, +0x1.2C0C36p+1f,
			-0x1.3714ECp+0f, +0x1.FD42EEp-1f,
			-0x1.274494p+0f, -0x1.3A1E10p+2f,
			-0x1.8CCCC6p+1f, -0x1.2B8944p-1f,
			+0x1.BD06B2p-1f, -0x1.1ED11Ap-2f,
			+0x1.590E08p-1f, +0x1.B76E50p-4f,
			-0x1.F6FF44p-2f, -0x1.D1E062p-2f,
			-0x1.5E098Cp-6f, +0x1.4B63BCp-4f,
			-0x1.BF4402p-2f, +0x1.459CE0p+0f,
			+0x1.54DEF8p-1f, +0x1.11D13Cp-1f,
			-0x1.239C4Cp+1f, -0x1.5B311Ap+0f,
			+0x1.10FD92p-2f, +0x1.3FEC78p-1f,
			-0x1.4E9BDAp+0f, -0x1.95CFD4p-1f,
			-0x1.C005A6p-1f, +0x1.A752E0p-2f,
			+0x1.05DA20p+0f, -0x1.F777BAp-1f,
			-0x1.0832AEp+1f, -0x1.E60424p-4f,
			+0x1.6C27D2p+0f, -0x1.3CC866p-1f,
			-0x1.2327C6p-4f, +0x1.148BD2p+0f,
			+0x1.6B39F0p+0f, +0x1.8283AAp+0f,
			+0x1.42E792p+1f, -0x1.A28680p+1f,
			+0x1.12F404p-1f, -0x1.136422p+1f,
			-0x1.9F1C9Ep-2f, -0x1.509E76p+1f,
			+0x1.06FB36p-5f, -0x1.574CE6p+0f,
			-0x1.2F8ED4p+0f, -0x1.037EE4p+0f,
			+0x1.C45A1Ep+1f, +0x1.72A882p-4f,
			-0x1.63C286p-2f, -0x1.137090p+1f,
			+0x1.C5980Ap+1f, +0x1.9BE3F2p-2f,
			-0x1.27CA92p+0f, +0x1.3E61BAp+1f,
			-0x1.0FCA54p+1f, -0x1.A01F30p+1f
		};
	}
}
