#ifndef ERROR_HPP_BA8CF134_C99C_4A3A_BEB4_E688A62EF47D
#define ERROR_HPP_BA8CF134_C99C_4A3A_BEB4_E688A62EF47D
// SOLUTION_TAGS: vulkan-(ex-[^1]|cw-.)

#include <string>
#include <exception>

namespace labutils
{
	// Class used for exceptions. Unlike e.g. std::runtime_error, which only
	// accepts a "fixed" string, Error provides std::printf()-like formatting.
	// Example:
	//
	//	throw Error( "vkCreateInstance() returned %s", to_string(result).c_str() );
	//
	class Error : public std::exception
	{
		public:
			explicit Error( char const*, ... );

		public:
			char const* what() const noexcept override;

		private:
			std::string mMsg;
	};
}

#endif // ERROR_HPP_BA8CF134_C99C_4A3A_BEB4_E688A62EF47D
